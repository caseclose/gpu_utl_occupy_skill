#!/usr/bin/env python3
"""occupy_gpu.py — CUDA / MLU / NPU 占卡工具

用法:
    occupy_gpu.py <gpus> [--backend cuda|mlu|npu] [--pct 百分比] [--period 秒] [--size N] [-f]

示例:
    occupy_gpu.py 0,1,3          # 把 0/1/3 卡利用率拉满(100%)
    occupy_gpu.py 7 --pct 5      # 卡 7 长期平均约 5% 利用率
    occupy_gpu.py all --pct 20   # 所有卡平均约 20%
    occupy_gpu.py 7 --pct 5 -f   # 前台运行, Ctrl+C 停止
    occupy_gpu.py --backend npu 0 --pct 20 -f   # 昇腾 NPU (寒武纪用 mlu)
    occupy_gpu.py --status       # 查看运行中的占卡进程与各卡利用率

停止:
    kill_occupy                  # 杀掉所有后台占卡进程(优雅退出)
    前台模式直接 Ctrl+C
"""
import argparse
import importlib
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import signal
import subprocess
import sys
import time


LABELS = {"cuda": "GPU", "mlu": "MLU", "npu": "NPU"}
STATUS_COMMANDS = {
    "cuda": ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv"],
    "mlu": ["cnmon"],
    "npu": ["npu-smi", "info"],
}


def worker(card: int, size: int, pct: float, period: float, backend: str):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C 由主进程统一清理
    if backend == "cuda":
        # 保持原有 CUDA 卡号语义: 单卡映射到进程内 cuda:0。
        os.environ["CUDA_VISIBLE_DEVICES"] = str(card)
    import torch

    if backend != "cuda":
        importlib.import_module(f"torch_{backend}")  # registers torch.mlu / torch.npu
        runtime = getattr(torch, backend)
        runtime.set_device(card)  # events and synchronize use the current device
        device = torch.device(f"{backend}:{card}")
        dtype = torch.float16
    else:
        runtime = torch.cuda
        device = torch.device("cuda:0")
        dtype = torch.float32
    tag = f"[{LABELS[backend]} {card}]"

    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    out = torch.empty_like(a)
    torch.matmul(a, b, out=out)  # 首次算子执行成功后再报告 running
    runtime.synchronize()
    print(f"{tag} running, pid={os.getpid()}, "
          f"tensor={size}x{size} {dtype}, target={pct}%", flush=True)

    if pct >= 100:
        # 满载模式: 循环小 kernel 把利用率顶满(与 v1 行为一致)
        # 不重新赋值给 a, 避免数值发散
        while True:
            for _ in range(20):
                torch.matmul(a, b, out=out)
            runtime.synchronize()
        return

    # 限流模式: 每 period 秒一个周期, 先让 GPU 连续忙
    # period*pct 时长(用设备 event 实测忙时, 不用 CPU 墙钟),
    # 再睡到周期结束 → 长期平均利用率 ≈ pct。
    # 忙时必须是"连续块"而不是碎片 kernel, 否则驱动利用率计数器
    # 采样不到, nvidia-smi 会显示 0%。
    target_busy = period * pct / 100.0
    ev0, ev1 = runtime.Event(enable_timing=True), runtime.Event(enable_timing=True)

    # ---- 校准: 单 kernel GPU 耗时(预热 8 次取中位数) ----
    ts = []
    for _ in range(8):
        ev0.record(); torch.matmul(a, b, out=out); ev1.record()
        runtime.synchronize()
        ts.append(ev0.elapsed_time(ev1) / 1000.0)
    ts.sort()
    t_kernel = max(ts[len(ts)//2], 1e-6)
    print(f"{tag} calibrated kernel={t_kernel*1000:.2f}ms", flush=True)

    # ---- 连续块 + 平滑闭环 ----
    # 每周期把忙时打成一个连续块(碎片化会被 nvidia-smi 漏计)。
    # 块速率用"连续排队"实测(与同步校准值不同: 流水线 overlap 使
    # 连续执行的单 kernel 更快), 避免按同步值排队导致 2 倍过冲。
    # 闭环: 用上一周期 busy 误差修正下一周期预算, 但修正量限幅
    # ±50%, 防止极端振荡; 长期平均收敛到 target_busy / period = pct。
    N_CAL = 128
    ev0.record()
    for _ in range(N_CAL):
        torch.matmul(a, b, out=out)
    ev1.record()
    runtime.synchronize()
    t_per_k = max(ev0.elapsed_time(ev1) / 1000.0 / N_CAL, 1e-6)
    print(f"{tag} pipelined kernel={t_per_k*1000:.2f}ms", flush=True)

    budget = target_busy
    log_n = 0
    while True:
        t0 = time.monotonic()
        n_burst = max(1, round(budget / t_per_k))
        ev0.record()
        for _ in range(n_burst):
            torch.matmul(a, b, out=out)
        ev1.record()
        runtime.synchronize()
        busy = ev0.elapsed_time(ev1) / 1000.0
        err = busy - target_busy              # 秒; 正=过冲 负=欠账
        corr = max(-0.5 * target_busy, min(0.5 * target_busy, -err))
        budget = max(2 * t_per_k, target_busy + corr)
        if log_n < 5 or log_n % 50 == 0:
            print(f"{tag} cycle: busy={busy*1000:.0f}ms "
                  f"err={err*1000:+.0f}ms budget_next={budget*1000:.0f}ms",
                  flush=True)
        log_n += 1
        time.sleep(max(0.0, period - (time.monotonic() - t0)))


def visible_device_count(backend):
    # Do not initialize an accelerator runtime in a process that will fork.
    probe = subprocess.run(
        [sys.executable, "-c", "import importlib, sys, torch; b = sys.argv[1]; "
         "importlib.import_module('torch_' + b); print(getattr(torch, b).device_count())", backend],
        capture_output=True, text=True, timeout=60)
    if probe.returncode:
        detail = probe.stderr.strip().splitlines()
        raise ValueError(f"无法探测 {LABELS[backend]}: 请检查驱动以及当前 Python 的 torch / torch_{backend} 版本"
                         + (f" ({detail[-1]})" if detail else ""))
    lines = probe.stdout.strip().splitlines()
    if not lines or not lines[-1].isdigit():
        raise ValueError(f"无法读取 torch.{backend}.device_count()")
    count = int(lines[-1])
    if not count:
        raise ValueError(f"当前 PyTorch 环境没有可见 {LABELS[backend]}")
    return count


def parse_gpus(spec: str, backend: str) -> list:
    count = visible_device_count(backend) if backend != "cuda" else None
    if spec.strip().lower() == "all":
        if count is not None:
            return list(range(count))
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
        cards = [int(x) for x in out.decode().split()]
        if not cards:
            raise ValueError("nvidia-smi 未列出 GPU")
        return cards
    gpus = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"非法设备编号: {part!r}")
        card = int(part)
        if count is not None and card >= count:
            raise ValueError(f"{LABELS[backend]} {card} 不可见; 可用逻辑编号为 0 到 {count - 1}")
        if card not in gpus:
            gpus.append(card)
    if not gpus:
        raise ValueError("没有指定任何设备")
    return gpus


def show_status(backend):
    r = subprocess.run(["pgrep", "-af", "occupy_gpu.py"],
                       capture_output=True, text=True)
    rows = [row for row in r.stdout.splitlines() if "--status" not in row.split()]
    if rows:
        print("运行中的占卡进程:")
        print("\n".join(rows))
    else:
        print("没有运行中的 occupy_gpu 进程")
    print()
    print("各卡当前状态:", flush=True)
    subprocess.run(STATUS_COMMANDS[backend], check=True)


def daemonize(log_path: str):
    """双重 fork + setsid 进入后台,标准输出/错误重定向到日志文件。"""
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, sys.stdin.fileno())
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, sys.stdout.fileno())
    os.dup2(fd, sys.stderr.fileno())


def main():
    p = argparse.ArgumentParser(
        prog="occupy_gpu",
        description="占卡:默认算力利用率拉满;--pct 指定长期平均利用率(默认后台运行)")
    p.add_argument("gpus", nargs="?", help="设备编号,逗号分隔,如 0,1,3;或 all")
    p.add_argument("--backend", choices=LABELS, default="cuda",
                   help="设备后端 (默认 cuda; MLU / NPU 使用 PyTorch 可见的逻辑编号)")
    p.add_argument("--pct", type=float, default=100.0,
                   help="目标长期平均利用率%% (默认 100 即满载)")
    p.add_argument("--period", type=float, default=10.0,
                   help="占空比周期,秒 (默认 10, 仅 pct<100 时生效; "
                        "周期长则忙块大)")
    p.add_argument("--size", type=int, default=2048,
                   help="矩阵边长,越大单 kernel 越重、显存越多 (默认 2048)")
    p.add_argument("-f", "--foreground", action="store_true",
                   help="前台运行(默认后台),前台时可用 Ctrl+C 停止")
    p.add_argument("--status", action="store_true",
                   help="查看运行中的占卡进程与各卡利用率")
    args = p.parse_args()

    if args.status:
        try:
            show_status(args.backend)
        except (OSError, subprocess.SubprocessError) as e:
            p.exit(1, f"无法读取设备状态: {e}\n")
        return

    if not args.gpus:
        p.error("必须指定设备编号(或 all),除非使用 --status")

    if not (0 < args.pct <= 100):
        p.error("--pct 取值范围 (0, 100]")
    if not math.isfinite(args.period) or args.period <= 0:
        p.error("--period 必须为正数")

    if args.size <= 0:
        p.error("--size 必须为正整数")
    try:
        gpus = parse_gpus(args.gpus, args.backend)
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        p.error(str(e))

    label = LABELS[args.backend]
    if not args.foreground:
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "occupy_gpu.log")
        print(f"正在后台启动,目标 {label}: {gpus}, 目标利用率 {args.pct}%\n"
              f"日志: {log_path}\n"
              f"停止: kill_occupy (或 occupy_gpu.py --backend {args.backend} --status 查看状态)", flush=True)
        daemonize(log_path)

    print(f"占用 {label}: {gpus} (Ctrl+C 停止)" if args.foreground
          else f"占用 {label}: {gpus}")
    ctx = mp.get_context("fork")  # 继承启动命令行, 供 kill_occupy 识别; 主进程未加载 torch
    procs = [ctx.Process(target=worker, args=(g, args.size, args.pct, args.period, args.backend),
                        daemon=True)
             for g in gpus]
    for proc in procs:
        proc.start()

    def shutdown(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        # Workers run until stopped. One unexpected exit should release all cards.
        wait([proc.sentinel for proc in procs])
        raise SystemExit(1)
    finally:
        print(f"正在释放所有 {label} ...", flush=True)
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join()


if __name__ == "__main__":
    main()
