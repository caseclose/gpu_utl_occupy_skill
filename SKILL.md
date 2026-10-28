---
name: gpu-occupy
description: GPU 占用工具。把指定 GPU 的算力利用率拉满(100%)或用 --pct 限流到指定长期平均利用率(如 80%/5%),用于占住机器防止资源回收、保持 GPU 活跃等场景。当用户提到"占卡"、"占 GPU"、"GPU 占用率"、"保持 GPU 利用率"、"防止机器被回收"等意图时使用本 skill。
---

# GPU 占用 Skill

把 GPU 算力利用率拉满或限流到指定百分比(长期平均),支持多卡、后台运行与状态查询。

## 环境要求

- Python 3
- PyTorch(CUDA 版):`pip install torch`
- NVIDIA GPU + nvidia-smi

## 快速开始

```bash
cd <skill 目录>/scripts
chmod +x occupy_gpu kill_occupy

./occupy_gpu 0,1,3          # 把 0/1/3 卡利用率拉满(100%)
./occupy_gpu 7 --pct 5      # 卡 7 长期平均约 5% 利用率
./occupy_gpu all --pct 80   # 所有卡平均约 80%
./occupy_gpu 7 --pct 5 -f   # 前台运行, Ctrl+C 停止
./occupy_gpu.py --status    # 查看运行中的占卡进程与各卡利用率
./kill_occupy               # 停止所有后台占卡进程
```

默认**后台运行**(双 fork 守护进程),日志追加在脚本同目录 `occupy_gpu.log`。

## 参数说明

| 参数 | 说明 | 默认 |
|---|---|---|
| `gpus` | GPU 编号,逗号分隔(如 `0,1,3`),或 `all` | 必填(除非 --status) |
| `--pct` | 目标长期平均利用率 %(取值 (0, 100]) | 100(满载) |
| `--period` | 占空比周期,秒。周期长则忙块大,nvidia-smi 读数更稳 | 10 |
| `--size` | 矩阵边长,越大单 kernel 越重、显存越多 | 2048 |
| `-f/--foreground` | 前台运行(默认后台) | 后台 |
| `--status` | 查看运行中进程与各卡利用率 | - |

## 工作原理(限流模式)

每 `period` 秒一个周期:先让 GPU **连续忙** `period × pct` 时长,再睡到周期结束,长期平均利用率收敛到 `pct`。

四个关键实现细节(踩过的坑):

1. **忙时必须是连续块**:碎片化的短 kernel(逐个同步发小 matmul)驱动利用率计数器采样不到,nvidia-smi 会恒显 0%。一个周期内所有 kernel 连续排队、末尾一次同步。
2. **用 CUDA event 计时**:CPU 提交完 kernel 就返回,真正的 GPU 执行时长与 CPU 墙钟差得很远,必须按 event 实测忙时闭环控制。
3. **burst 数用"流水线速率"估算**:同步执行的单 kernel 耗时和连续排队时不同(流水线 overlap),启动时先排 128 个 kernel 实测流水线速率,再定每周期 kernel 数,避免 2 倍过冲。
4. **闭环修正限幅**:用上一周期忙时误差修正下一周期预算,修正量限幅 ±50%,防止在"几乎不跑"和"连跑数秒"之间振荡。

## 验证占用率

瞬时 `nvidia-smi` 读数会在忙窗口 ~100%、闲窗口 0% 之间跳(默认 period=10s 即每 10 秒一个约 8 秒的忙块,这是采样粒度下的正常现象),验证请看长期均值:

```bash
nvidia-smi dmon -i <GPU号> -s u -d 1 -c 300   # 5 分钟均值应接近 --pct
```

或查看日志中的闭环输出(`busy/period ≈ pct` 即达标):

```
[GPU 7] cycle: busy=7999ms err=-1ms budget_next=8001ms   # 8s/10s = 80%
```

## 注意事项

- 限流模式默认后台运行,日志追加在 `occupy_gpu.log`,先用 `--status` 确认再排查日志。
- **一台机器同时只跑一个实例**;重复启动会叠加进程(实际利用率超过目标),启动前先 `./kill_occupy`。
- 满载模式会让 GPU 持续高功耗运行,注意机器散热与电费预算。
- 结束任务用 `kill_occupy`(后台)或 Ctrl+C(前台),不要直接 `kill -9`(可能残留显存)。
- 每卡显存占用约 473 MiB(2048×2048 fp32 矩阵 ×3)。

## 文件说明

| 文件 | 说明 |
|---|---|
| `scripts/occupy_gpu.py` | 主脚本(满载 + 限流 + --status) |
| `scripts/occupy_gpu` | bash 包装,等价 `python3 occupy_gpu.py` |
| `scripts/kill_occupy` | 杀掉所有占卡进程(SIGTERM,5 秒后兜底 SIGKILL) |
