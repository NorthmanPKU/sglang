# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Request scheduler policy"""

import os
import random
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Union

import torch

from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    global_server_args_dict,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import TokenToKVPoolAllocator
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

# Clip the estimation of max_new_tokens for the request whose max_new_tokens is very large.
# This can prevent the server from being too conservative.
# Note that this only clips the estimation in the scheduler but does not change the stop
# condition. The request can still generate tokens until it hits the unclipped max_new_tokens.
CLIP_MAX_NEW_TOKENS_ESTIMATION = int(
    os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", "4096")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (against existing cache) less than this value,
# the scheduler runs the in-batch prefix caching check for this request.
# If we set it to -1, it means we disable in-batch prefix caching.
IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD", "32")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (within the waiting queue) larger than this value,
# the scheduler deprioritizes this request
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD", "32")
)

# 主要用于给waiting queue排序，waiting queue就是待处理的请求队列
class CacheAwarePolicy(Enum):
    """Scheduling policies that are aware of the tree cache."""

    LPM = "lpm"  # longest prefix match
    DFS_WEIGHT = "dfs-weight"  # depth-first search weighting


class CacheAgnosticPolicy(Enum):
    """Scheduling policies that are not aware of the tree cache."""

    FCFS = "fcfs"  # first come first serve
    LOF = "lof"  # longest output first
    RANDOM = "random"


class SchedulePolicy: # 默认使用了RadixCache作为waiting_queue_radix_tree用于任务调度
    Policy = Union[CacheAwarePolicy, CacheAgnosticPolicy]

    def __init__(self, policy: str, tree_cache: BasePrefixCache):
        self.policy = self._validate_and_adjust_policy(policy, tree_cache)
        self.tree_cache = tree_cache

        # It is used to find the matching prefix for in-batch prefix caching.
        self.waiting_queue_radix_tree = RadixCache(
            req_to_token_pool=None, token_to_kv_pool_allocator=None, disable=False
        )

    # 计算优先级，最重要的接口函数
    def calc_priority(self, waiting_queue: List[Req]) -> bool:
        policy = self._determine_active_policy(waiting_queue)

        prefix_computed = False
        if isinstance(policy, CacheAwarePolicy):
            prefix_computed = True
            temporary_deprioritized = self._compute_prefix_matches(
                waiting_queue, policy
            )
            if policy == CacheAwarePolicy.LPM:
                SchedulePolicy._sort_by_longest_prefix(
                    waiting_queue, temporary_deprioritized
                )
            elif policy == CacheAwarePolicy.DFS_WEIGHT:
                SchedulePolicy._sort_by_dfs_weight(waiting_queue, self.tree_cache)
            else:
                raise ValueError(f"Unknown CacheAware Policy: {policy=}")
        else:
            if policy == CacheAgnosticPolicy.FCFS:
                pass
            elif policy == CacheAgnosticPolicy.LOF:
                SchedulePolicy._sort_by_longest_output(waiting_queue)
            elif policy == CacheAgnosticPolicy.RANDOM:
                SchedulePolicy._sort_randomly(waiting_queue)
            else:
                raise ValueError(f"Unknown CacheAgnostic Policy: {policy=}")

        return prefix_computed

    # _determine_active_policy 中如果发现等待队列太长且默认采用的是LPM（最长前缀匹配），则换成FCFS
    def _determine_active_policy(self, waiting_queue: List[Req]) -> Policy:
        if len(waiting_queue) > 128 and self.policy == CacheAwarePolicy.LPM:
            # Turn off the expensive prefix matching and sorting when the #queue is large.
            return CacheAgnosticPolicy.FCFS
        return self.policy

    def _validate_and_adjust_policy(
        self, policy: str, tree_cache: BasePrefixCache
    ) -> Policy:
        """
        Validates the policy and adjusts it if necessary based on tree cache settings.
        """
        try:
            policy_enum = CacheAwarePolicy(policy)
            if tree_cache.disable:
                # If tree_cache is disabled, using CacheAgnosticPolicy policy
                return CacheAgnosticPolicy.FCFS
            return policy_enum
        except ValueError:
            try:
                return CacheAgnosticPolicy(policy)
            except ValueError:
                raise ValueError(f"Unknown schedule_policy: {policy=}")

    def _compute_prefix_matches(
        self, waiting_queue: List[Req], policy: CacheAwarePolicy
    ) -> Set[int]:
        """
        Computes and caches the matching prefixes for requests in the waiting queue,
            and handles in-batch prefix caching logic.
        """
        temporary_deprioritized: Set[int] = set()
        self.waiting_queue_radix_tree.reset()

        for r in waiting_queue:
            prefix_ids = r.adjust_max_prefix_ids()

            # NOTE: the prefix_indices must always be aligned with last_node
            r.prefix_indices, r.last_node = self.tree_cache.match_prefix(
                rid=r.rid, key=prefix_ids
            )

            # NOTE(sang): This logic is for in-batch prefix caching;
            # If there are more than 1 request that have small matching prefix from
            # existing cache, but all those requests share the same prefix, we prefer
            # to schedule only one of them so that we can increase the cache hit rate.
            # We prefer to set IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD > 0 because too small
            # threshold means we cannot use in-batch prefix caching for short prefixes.
            # It is kind of common when the engine is long running (e.g., imagine the prefix "the").
            if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
                in_batch_matching_prefixes, _ = (
                    self.waiting_queue_radix_tree.match_prefix(
                        rid=r.rid, key=prefix_ids
                    )
                )
                if (
                    len(in_batch_matching_prefixes)
                    >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
                ):
                    temporary_deprioritized.add(r.rid)
                else:
                    # Insert with a dummy key
                    self.waiting_queue_radix_tree.insert(
                        prefix_ids, torch.empty(len(prefix_ids), dtype=torch.bool)
                    )
        return temporary_deprioritized

    @staticmethod
    def _sort_by_longest_prefix(
        waiting_queue: List[Req], temporary_deprioritized: Set[int]
    ) -> None:
        """Sorts the waiting queue based on the longest prefix match."""
        waiting_queue.sort(
            key=lambda r: (
                -len(r.prefix_indices)
                if r.rid not in temporary_deprioritized
                else float("inf")
            )
        )

    @staticmethod
    def _sort_by_dfs_weight(
        waiting_queue: List[Req], tree_cache: BasePrefixCache
    ) -> None:
        """Sorts the waiting queue based on a depth-first search weighting."""
        last_node_to_reqs = defaultdict(list)
        for req in waiting_queue:
            last_node_to_reqs[req.last_node].append(req)

        node_to_weight = defaultdict(int)
        for node in last_node_to_reqs:
            node_to_weight[node] = len(last_node_to_reqs[node])
        SchedulePolicy._calc_weight(tree_cache.root_node, node_to_weight)

        waiting_queue.clear()
        SchedulePolicy._get_dfs_priority(
            tree_cache.root_node,
            node_to_weight,
            last_node_to_reqs,
            waiting_queue,
        )

    @staticmethod
    def _sort_by_longest_output(waiting_queue: List[Req]) -> None:
        """Sorts the waiting queue based on the longest output (max_new_tokens)."""
        waiting_queue.sort(key=lambda x: -x.sampling_params.max_new_tokens)

    @staticmethod
    def _sort_randomly(waiting_queue: List[Req]) -> None:
        """Shuffles the waiting queue randomly."""
        random.shuffle(waiting_queue)

    @staticmethod
    def _calc_weight(cur_node: TreeNode, node_to_weight: Dict[TreeNode, int]) -> None:
        for child in cur_node.children.values():
            SchedulePolicy._calc_weight(child, node_to_weight)
            node_to_weight[cur_node] += node_to_weight[child]

    @staticmethod
    def _get_dfs_priority(
        cur_node: TreeNode,
        node_to_priority: Dict[TreeNode, int],
        last_node_to_reqs: Dict[TreeNode, List[Req]],
        q: List,
    ) -> None:
        childs = [child for child in cur_node.children.values()]
        childs.sort(key=lambda x: -node_to_priority[x])
        for child in childs:
            SchedulePolicy._get_dfs_priority(
                child, node_to_priority, last_node_to_reqs, q
            )
        q.extend(last_node_to_reqs[cur_node])


class AddReqResult(Enum):
    CONTINUE = auto()  # Continue to add requests
    NO_TOKEN = auto()  # No token left
    OTHER = auto()  # Other reasons to stop adding requests

# 决定了还能不能插入新请求
class PrefillAdder:
    def __init__(
        self,
        tree_cache: BasePrefixCache,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        running_batch: ScheduleBatch,
        new_token_ratio: float,
        rem_input_tokens: int,
        rem_chunk_tokens: Optional[int],
        mixed_with_decode_tokens: int = 0,
    ):
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.running_batch = running_batch
        self.new_token_ratio = new_token_ratio
        self.rem_input_tokens = rem_input_tokens - mixed_with_decode_tokens # 只包括prefill 的输入
        self.rem_chunk_tokens = rem_chunk_tokens # 一个chunk可以包含的token数
        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= mixed_with_decode_tokens

        self.rem_total_token_offset = mixed_with_decode_tokens
        self.cur_rem_token_offset = mixed_with_decode_tokens

        self.req_states = None
        self.can_run_list = []
        self.new_chunked_req = None
        self.log_hit_tokens = 0
        self.log_input_tokens = 0

        if running_batch is not None:
            self.rem_total_token_offset += sum(
                [
                    min(
                        (r.sampling_params.max_new_tokens - len(r.output_ids)),
                        CLIP_MAX_NEW_TOKENS_ESTIMATION,
                    )
                    * self.new_token_ratio
                    for r in running_batch.reqs
                ]
            )

    @property
    def rem_total_tokens(self): # 包括prefill和decoding 一共的上下文长度
        return (
            self.token_to_kv_pool_allocator.available_size()
            + self.tree_cache.evictable_size()
            - self.rem_total_token_offset
        )

    @property
    def cur_rem_tokens(self):
        return (
            self.token_to_kv_pool_allocator.available_size()
            + self.tree_cache.evictable_size()
            - self.cur_rem_token_offset
        )

    def budget_state(self):
        if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        if self.rem_input_tokens <= 0 or (
            self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0
        ):
            return AddReqResult.OTHER

        return AddReqResult.CONTINUE

    def _prefill_one_req(
        self, prefix_len: int, extend_input_len: int, max_new_tokens: int
    ):
        """
        更新批次的资源使用情况和日志记录。

        该函数在添加请求到批次时被调用，用于更新当前批次的资源消耗状态，
        并记录处理的token数。它确保批次处理不会超出可用资源，并为后续分析
        和调试提供数据支持。

        参数:
            prefix_len (int): 前缀长度，表示已缓存的token数。
            extend_input_len (int): 扩展输入长度，表示新输入的token数。
            max_new_tokens (int): 最大新生成token数，表示请求可能生成的token数。

        作用:
            - 更新总token偏移量和当前token偏移量。
            - 减少剩余输入token数和分块token数（如果启用）。
            - 记录命中token数和输入token数。
        """
        self.rem_total_token_offset += extend_input_len + max_new_tokens
        self.cur_rem_token_offset += extend_input_len
        self.rem_input_tokens -= extend_input_len
        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= extend_input_len

        self.log_hit_tokens += prefix_len
        self.log_input_tokens += extend_input_len

    def add_chunked_req(self, req: Req):
        """
        处理分块请求，将其添加到可运行列表中，并根据资源限制截断请求。

        该函数用于处理长序列请求，通过分块策略将请求分成可管理的部分。
        它根据剩余资源截断请求，并更新批次的资源使用情况。对于被截断的
        请求，函数返回请求对象以便后续继续处理。

        参数:
            req (Req): 请求对象，包含输入序列和采样参数。

        返回:
            Req: 如果请求被截断，返回请求对象；否则返回None表示请求已完整处理。

        作用:
            - 根据剩余分块token数截断请求。
            - 更新请求的扩展输入长度和填充ID数组。
            - 将请求添加到可运行列表。
            - 调用`_prefill_one_req`更新资源使用情况。
        """
        truncated = req.extend_input_len > self.rem_chunk_tokens
        req.extend_input_len = min(req.extend_input_len, self.rem_chunk_tokens)
        req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]
        self.can_run_list.append(req)
        self._prefill_one_req(
            0,
            req.extend_input_len,
            (
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS_ESTIMATION)
                if not truncated
                else 0
            ),
        )

        # Return if chunked prefill not finished
        return req if truncated else None

    @contextmanager
    def _lock_node(self, last_node: TreeNode):
        try:
            self.tree_cache.inc_lock_ref(last_node)
            yield None
        finally:
            self.tree_cache.dec_lock_ref(last_node)

    def add_one_req_ignore_eos(self, req: Req, has_chunked_req: bool):
        def add_req_state(r, insert_sort=False):
            new_token_ratio = (
                1.0 if r.sampling_params.ignore_eos else self.new_token_ratio
            )
            tokens_left = r.sampling_params.max_new_tokens * new_token_ratio - len(
                r.output_ids
            )
            tokens_occupied = len(r.origin_input_ids) + len(r.output_ids)

            if tokens_left > 0:
                if not insert_sort:
                    self.req_states.append((tokens_left, tokens_occupied))
                else:
                    i = 0
                    for i in range(len(self.req_states)):
                        if tokens_left <= self.req_states[i][0]:
                            break
                    self.req_states.insert(i, (tokens_left, tokens_occupied))

        if self.req_states is None:
            self.req_states = []
            add_req_state(req)
            if self.running_batch is not None:
                for r in self.running_batch.reqs:
                    add_req_state(r)
            for r in self.can_run_list:
                add_req_state(r)
            self.req_states.sort(key=lambda x: x[0])
        else:
            add_req_state(req, insert_sort=True)

        cur_rem_tokens = self.cur_rem_tokens - len(req.origin_input_ids)
        tokens_freed = 0
        for i, (tokens_left, tokens_occupied) in enumerate(self.req_states):
            decode_steps = (
                self.req_states[i + 1][0]
                if i + 1 < len(self.req_states)
                else tokens_left
            )
            bs = len(self.req_states) - i
            if cur_rem_tokens + tokens_freed - decode_steps * bs <= 0:
                return AddReqResult.NO_TOKEN
            tokens_freed += tokens_occupied

        if (
            self.rem_chunk_tokens is None  # chunked prefill is disabled
            or req.extend_input_len <= self.rem_chunk_tokens  # it is the last chunk
        ):
            # Non-chunked prefill
            self.can_run_list.append(req)
            self._prefill_one_req(
                0,
                req.extend_input_len,
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS_ESTIMATION),
            )
        else:
            if self.rem_chunk_tokens == 0:
                return AddReqResult.OTHER

            # Chunked prefill
            trunc_len = self.rem_chunk_tokens

            req.extend_input_len = trunc_len
            req.fill_ids = req.fill_ids[:trunc_len]
            self.can_run_list.append(req)
            self.new_chunked_req = req
            self._prefill_one_req(0, trunc_len, 0)

        return self.budget_state()

    def add_one_req(self, req: Req, has_chunked_req: bool):
        """
        尝试将单个请求添加到当前批次中，根据资源限制和分块策略决定如何处理。
        
        该函数检查请求所需的资源是否可以满足，并决定是完整处理请求、进行分块处理，
        还是需要延迟处理。对于需要分块处理的长序列请求，会截断输入并标记为分块请求，
        以便后续继续处理剩余部分。
        
        参数:
            req (Req): 待添加的请求对象，包含输入序列和采样参数。
            has_chunked_req (bool): 当前批次中是否已经存在分块请求，影响资源分配决策。
        
        返回:
            AddReqResult: 添加结果枚举值，可能为以下之一:
                - CONTINUE: 可以继续添加更多请求
                - NO_TOKEN: 没有足够的token资源
                - OTHER: 其他原因导致无法添加
        
        注意:
            1. 对于设置了ignore_eos且不使用树缓存的请求，使用特殊处理路径。
            2. 函数会锁定请求的last_node以确保并发安全。
            3. 当资源不足以完整处理但可以分块处理时，会截断输入并创建分块请求。
            4. 分块处理会更新req的extend_input_len和fill_ids，仅处理一部分输入。
        """
        # 特殊情况：如果请求设置了忽略EOS标记且树缓存禁用，使用专门的处理函数
        if req.sampling_params.ignore_eos and self.tree_cache.disable:
            return self.add_one_req_ignore_eos(req, has_chunked_req)

        # 计算请求的总token数（输入长度+估计的生成长度）
        # CLIP_MAX_NEW_TOKENS_ESTIMATION是一个常量，用于限制估计的生成长度上限
        total_tokens = req.extend_input_len + min(
            req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS_ESTIMATION
        )
        input_tokens = req.extend_input_len
        # 获取前缀长度（已缓存部分的长度）
        prefix_len = len(req.prefix_indices)

        # 如果请求所需的token数超过了当前可用的token总数，返回NO_TOKEN
        if total_tokens >= self.rem_total_tokens:
            return AddReqResult.NO_TOKEN

        # 如果请求的输入token数超过了当前可用的输入token总数，并且can_run_list不为空，返回OTHER
        if input_tokens > self.rem_input_tokens and len(self.can_run_list) != 0:
            return AddReqResult.OTHER

        # 使用上下文管理器锁定请求的last_node，确保并发安全
        with self._lock_node(req.last_node):
            if total_tokens > self.rem_total_tokens:
                return AddReqResult.NO_TOKEN

            if self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
                # Non-chunked prefill
                self.can_run_list.append(req)
                self.tree_cache.inc_lock_ref(req.last_node)
                self._prefill_one_req(
                    prefix_len,
                    input_tokens,
                    min(
                        req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKENS_ESTIMATION,
                    ),
                )
            else:
                if self.rem_chunk_tokens == 0:
                    return AddReqResult.OTHER

                # Chunked prefill
                trunc_len = self.rem_chunk_tokens

                req.extend_input_len = trunc_len
                req.fill_ids = req.fill_ids[: len(req.prefix_indices) + trunc_len]

                self.can_run_list.append(req)
                self.new_chunked_req = req
                self.tree_cache.inc_lock_ref(req.last_node)
                self._prefill_one_req(prefix_len, trunc_len, 0)

        return self.budget_state()
