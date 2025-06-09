#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shlex
import subprocess
import time
from pathlib import Path

# ======== 可按需修改 ========
MODEL_PATH = "/dev/shm/DeepSeek-V3-0324/"
RESULT_FILE = Path("bench_results.txt")
READY_PHRASE = "The server is fired up and ready to roll!"
# ===========================

server_cmds = {
    "flashinfer": (
        f"SGLANG_ENABLE_FLASHINFER_GEMM=1 "
        f"python3 -m sglang.launch_server "
        f"--model-path {MODEL_PATH} --tp 8 --trust-remote "
        f"--attention-backend flashinfer --disable-radix"
    ),
    # "triton": (
    #     f"SGLANG_ENABLE_FLASHINFER_GEMM=1 "
    #     f"python3 -m sglang.launch_server "
    #     f"--model-path {MODEL_PATH} --tp 8 --trust-remote "
    #     f"--attention-backend triton"
    # ),
}

client_cmds = {
    "rand256x32": (
        "python3 -m sglang.bench_serving --backend sglang-oai "
        "--dataset-name random --random-input-len 1000 "
        "--random-output-len 1000 --random-range-ratio 1 "
        "--num-prompts 256 --max-concurrency 32"
    ),
    "rand128x16": (
        "python3 -m sglang.bench_serving --backend sglang-oai "
        "--dataset-name random --random-input-len 1000 "
        "--random-output-len 1000 --random-range-ratio 1 "
        "--num-prompts 128 --max-concurrency 16"
    ),
}


def log(msg: str) -> None:
    """同时打印到终端并追加写入结果文件。"""
    print(msg)
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run():
    # 清空旧结果
    RESULT_FILE.unlink(missing_ok=True)

    # 针对每一个 server × client 的组合
    for s_name, s_cmd in server_cmds.items():
        for c_name, c_cmd in client_cmds.items():
            log("=" * 100)
            log(f"[组合] SERVER={s_name}  CLIENT={c_name}")
            log("=" * 100)

            # 1) 启动 server
            log(f"[SERVER] 启动 {s_name}: {s_cmd}")
            server_proc = subprocess.Popen(
                s_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # 2) 等待 server 就绪
            while True:
                line = server_proc.stdout.readline()
                if not line:
                    # 如果进程意外退出
                    if server_proc.poll() is not None:
                        raise RuntimeError(f"Server {s_name} 启动失败，进程已退出！")
                    time.sleep(0.2)
                    continue
                print(f"[{s_name}] {line}", end="")
                if READY_PHRASE in line:
                    log(f"[SERVER] {s_name} 就绪检测到：{READY_PHRASE}")
                    break

            # 3) 运行 client 并捕获输出
            log(f"[CLIENT] 运行 {c_name}: {c_cmd}")
            result = subprocess.run(
                shlex.split(c_cmd),
                capture_output=True,
                text=True,
            )

            # 4) 写入结果文件
            header = (
                f"\n\n{'#' * 80}\n"
                f"SERVER: {s_name}\n"
                f"CLIENT: {c_name}\n"
                f"COMMAND: {c_cmd}\n"
                f"{'#' * 80}\n"
            )
            with RESULT_FILE.open("a", encoding="utf-8") as f:
                f.write(header)
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n[STDERR]\n")
                    f.write(result.stderr)

            log(f"[CLIENT] {c_name} 完成，退出码 {result.returncode}")

            # 5) 关闭 server
            log(f"[SERVER] 关闭 {s_name}")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=30)
                log(f"[SERVER] {s_name} 已优雅退出")
            except subprocess.TimeoutExpired:
                server_proc.kill()
                log(f"[SERVER] {s_name} 强制杀死")

    log("\n所有组合测试完成！结果保存在：{}".format(RESULT_FILE.resolve()))


if __name__ == "__main__":
    run()
