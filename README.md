# gpu_utl_occupy_skill

用矩阵乘法在指定的 NVIDIA GPU、寒武纪 MLU 或华为昇腾 NPU 上产生计算负载。可以满载运行，也可以用 `--pct` 指定目标长期平均利用率。支持独立命令行和 [Claude / CodeBuddy Skill](SKILL.md)。

## 快速开始

在目标 Linux 服务器上，激活对应厂商的 Python 环境后运行：

```bash
cd scripts
./occupy_gpu 0,1 --pct 80              # 原有用法：默认 CUDA
./occupy_gpu --backend mlu 0 --pct 20 -f  # 寒武纪，前台运行
./occupy_gpu --backend npu 0 --pct 20 -f  # 昇腾，前台运行
```

以上为独立示例，选择符合硬件的一条。前台用 Ctrl+C 停止本次运行。省略 `-f` 时在后台运行，日志追加到 `scripts/occupy_gpu.log`。后台命令返回只表示已发起启动；检查各卡的 `running` 和后续日志确认运行情况。原有 `./kill_occupy` 会按脚本名停止匹配的占卡进程，**不区分后端或卡号**。

## 环境和编号

| `--backend` | 运行依赖 | `--status` 的设备数据来源 | 命令行卡号 |
|---|---|---|---|
| `cuda`（默认） | NVIDIA 驱动、CUDA 版 PyTorch | `nvidia-smi` | 沿用原有 CUDA 编号；`all` 从 nvidia-smi 枚举 |
| `mlu` | 寒武纪驱动、Neuware、[配套的 PyTorch 和 torch_mlu](https://github.com/Cambricon/torch_mlu) | `cnmon` | 当前 PyTorch 可见的 MLU 逻辑编号 |
| `npu` | 昇腾驱动、CANN、[配套的 PyTorch 和 torch_npu](https://github.com/Ascend/pytorch) | `npu-smi info` | 当前 PyTorch 可见的 NPU 逻辑编号 |

MLU / NPU 继承当前进程的可见设备设置，不改写映射。例如容器只暴露一张卡时，应传 `0`，即使宿主机管理工具给这张卡的编号不是 0。`all` 只选当前后端的可见设备；它不检查设备是否已有作业。查询 MLU / NPU 的状态需要管理工具，但运行负载不依赖它。

## 参数

| 参数 | 含义 |
|---|---|
| `--backend cuda\|mlu\|npu` | 默认 CUDA，不自动切换硬件 |
| `gpus` | 逗号分隔的编号（如 `0,2`）或 `all`；除 `--status` 外必填 |
| `--pct P` | 目标长期平均利用率，范围 (0, 100]，默认 100（满载） |
| `--period S` | 忙闲周期，默认 10 秒；仅在限流时生效 |
| `--size N` | 方阵边长，默认 2048。CUDA 使用 FP32，MLU / NPU 使用 FP16 |
| `-f` / `--foreground` | 前台运行；省略时使用后台模式 |
| `--status` | 列出匹配的工具进程，并显示所选后端的设备状态 |

## 在目标机器上验证

例如验证昇腾卡 0（寒武纪把 `npu` 换成 `mlu`）：

```bash
./occupy_gpu --backend npu --status
./occupy_gpu --backend npu 0 --size 1024 --pct 20 --period 5 -f
```

在另一个终端运行 `npu-smi info`（MLU 用 `cnmon`），观察多个完整周期。20% 配 5 秒周期时，目标是每周期约 1 秒忙、4 秒闲；瞬时读数会波动。日志中的 `cycle: busy=...ms` 是设备事件测得的忙窗口，和厂商监控的整卡利用率是两个指标，不保证数值相同。Ctrl+C 后再查状态，确认这次运行的进程已经退出。

NVIDIA 可用 `nvidia-smi dmon -i 0 -s u -d 1 -c 300` 连续采样 5 分钟，取利用率列的均值。

满载模式持续提交矩阵乘法。限流模式先连续提交 128 次矩阵乘法，估算每次用时，再按目标忙时安排每个周期的计算量；根据上个周期的事件耗时修正下个周期的预算，修正不超过目标忙时的 ±50%。重复启动会叠加负载，启动前请先确认目标卡可用。

无硬件测试：在仓库根目录运行 `python3 -m unittest discover -s tests`。这些测试验证后端选择、卡号和运行流程；利用率和具体型号兼容性需要在真机上测量。
