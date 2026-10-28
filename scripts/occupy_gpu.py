#!/usr/bin/env python3
"""occupy_gpu.py — GPU 占卡工具 v2

用法:
    occupy_gpu.py <gpus> [--pct 百分比] [--period 秒] [--size N] [-f]

示例:
    occupy_gpu.py 0,1,3          # 把 0/1/3 卡利用率拉满(100%)
    occupy_gpu.py 7 --pct 5      # 卡 7 长期平均约 5% 利用率
    occupy_gpu.py all --pct 20   # 所有卡平均约 20%
    occupy_gpu.py 7 --pct 5 -f   # 前台运行, Ctrl+C 停止
    occupy_gpu.py --status       # 查看运行中的占卡进程与各卡利用率

停止:
    kill_occupy                  # 杀掉所有后台占卡进程(优雅退出)
    前台模式直接 Ctrl+C
"""
import argparse
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time


def worker(gpu_id: int, size: int, pct: float, period: float):
    # 必须在 import torch 之前设置,把物理卡号映射为该进程内的 cuda:0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch

    device = torch.device("cuda:0")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    out = torch.empty(size, size, device=device)
    torch.cuda.synchronize()
    print(f"[GPU {gpu_id}] running, pid={os.getpid()}, "
          f"tensor={size}x{size} fp32, target={pct}%", flush=True)

    if pct >= 100:
        # 满载模式: 循环小 kernel 把利用率顶满(与 v1 行为一致)
        # 不重新赋值给 a, 避免数值发散
        while True:
            for _ in range(20):
                torch.matmul(a, b, out=out)
            torch.cuda.synchronize()
        return

    # 限流模式: 每 period 秒一个周期, 先让 GPU 连续忙
    # period*pct 时长(用 CUDA event 实测忙时, 不用 CPU 墙钟),
    # 再睡到周期结束 → 长期平均利用率 ≈ pct。
    # 忙时必须是"连续块"而不是碎片 kernel, 否则驱动利用率计数器
    # 采样不到, nvidia-smi 会显示 0%。
    target_busy = period * pct / 100.0
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)

    # ---- 校准: 单 kernel GPU 耗时(预热 8 次取中位数) ----
    ts = []
    for _ in range(8):
        ev0.record(); torch.matmul(a, b, out=out); ev1.record()
        torch.cuda.synchronize()
        ts.append(ev0.elapsed_time(ev1) / 1000.0)
    ts.sort()
    t_kernel = max(ts[len(ts)//2], 1e-6)
    print(f"[GPU {gpu_id}] calibrated kernel={t_kernel*1000:.2f}ms", flush=True)

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
    torch.cuda.synchronize()
    t_per_k = max(ev0.elapsed_time(ev1) / 1000.0 / N_CAL, 1e-6)
    print(f"[GPU {gpu_id}] pipelined kernel={t_per_k*1000:.2f}ms", flush=True)

    budget = target_busy
    log_n = 0
    while True:
        t0 = time.monotonic()
        n_burst = max(1, round(budget / t_per_k))
        ev0.record()
        for _ in range(n_burst):
            torch.matmul(a, b, out=out)
        ev1.record()
        torch.cuda.synchronize()
        busy = ev0.elapsed_time(ev1) / 1000.0
        err = busy - target_busy              # 秒; 正=过冲 负=欠账
        corr = max(-0.5 * target_busy, min(0.5 * target_busy, -err))
        budget = max(2 * t_per_k, target_busy + corr)
        if log_n < 5 or log_n % 50 == 0:
            print(f"[GPU {gpu_id}] cycle: busy={busy*1000:.0f}ms "
                  f"err={err*1000:+.0f}ms budget_next={budget*1000:.0f}ms",
                  flush=True)
        log_n += 1
        time.sleep(max(0.0, period - (time.monotonic() - t0)))


def parse_gpus(spec: str) -> list:
    if spec.strip().lower() == "all":
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
        return [int(x) for x in out.decode().split()]
    gpus = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"非法 GPU 编号: {part!r}")
        gpus.append(int(part))
    if not gpus:
        raise ValueError("没有指定任何 GPU")
    return gpus


def show_status():
    r = subprocess.run(["pgrep", "-af", "occupy_gpu.py"],
                       capture_output=True, text=True)
    if r.stdout.strip():
        print("运行中的占卡进程:")
        print(r.stdout.rstrip())
    else:
        print("没有运行中的 occupy_gpu 进程")
    print()
    print("各卡当前状态:")
    subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                    "--format=csv"])


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
    p.add_argument("gpus", nargs="?", help="GPU 编号,逗号分隔,如 0,1,3;或 all")
    p.add_argument("--pct", type=float, default=100.0,
                   help="目标长期平均利用率%% (默认 100 即满载)")
    p.add_argument("--period", type=float, default=10.0,
                   help="占空比周期,秒 (默认 10, 仅 pct<100 时生效; "
                        "周期长则忙块大, nvidia-smi 读数更稳)")
    p.add_argument("--size", type=int, default=2048,
                   help="矩阵边长,越大单 kernel 越重、显存越多 (默认 2048)")
    p.add_argument("-f", "--foreground", action="store_true",
                   help="前台运行(默认后台),前台时可用 Ctrl+C 停止")
    p.add_argument("--status", action="store_true",
                   help="查看运行中的占卡进程与各卡利用率")
    args = p.parse_args()

    if args.status:
        show_status()
        return

    if not args.gpus:
        p.error("必须指定 GPU 编号(或 all),除非使用 --status")

    try:
        gpus = parse_gpus(args.gpus)
    except ValueError as e:
        p.error(str(e))

    if not (0 < args.pct <= 100):
        p.error("--pct 取值范围 (0, 100]")
    if args.period <= 0:
        p.error("--period 必须为正数")

    if not args.foreground:
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "occupy_gpu.log")
        print(f"已后台启动,占用 GPU: {gpus}, 目标利用率 {args.pct}%\n"
              f"日志: {log_path}\n"
              f"停止: kill_occupy (或 occupy_gpu.py --status 查看状态)", flush=True)
        daemonize(log_path)

    print(f"占用 GPU: {gpus} (Ctrl+C 停止)" if args.foreground
          else f"占用 GPU: {gpus}")
    procs = [mp.Process(target=worker, args=(g, args.size, args.pct, args.period),
                        daemon=True)
             for g in gpus]
    for proc in procs:
        proc.start()

    def shutdown(*_):
        print("\n正在释放所有 GPU ...")
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.join()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()
