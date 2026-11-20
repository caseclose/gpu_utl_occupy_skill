---
name: gpu-occupy
description: 在指定 NVIDIA GPU、寒武纪 MLU 或华为昇腾 NPU 上产生矩阵运算负载，支持多卡和目标平均利用率。用户要求占卡、调整负载、查看设备状态或使用 NVIDIA 网页看板时使用。
---

# 加速卡占用

在本 Skill 的 `scripts/` 目录运行命令。根据用户指定的硬件选择后端：

| 硬件 | 查看状态 | 示例：在卡 0 上前台运行，目标平均 20% |
|---|---|---|
| NVIDIA GPU | `./occupy_gpu --status` | `./occupy_gpu 0 --pct 20 -f` |
| 寒武纪 MLU | `./occupy_gpu --backend mlu --status` | `./occupy_gpu --backend mlu 0 --pct 20 -f` |
| 华为昇腾 NPU | `./occupy_gpu --backend npu --status` | `./occupy_gpu --backend npu 0 --pct 20 -f` |

- MLU / NPU 的编号是当前 PyTorch 环境可见的逻辑编号；管理工具可能显示宿主机物理编号。`0,1` 选两卡，`all` 选该后端的全部可见卡，不检查卡是否空闲。先查看状态，不要把请求的单卡扩大成全部卡。
- 默认目标为 100%。命令行和网页看板按卡执行同一负载策略。目标 T ≤ 20% 时使用固定低占空比。T > 20% 时，在 T ± 8 个百分点与 0%–100% 的交集内选值，分段保持并平滑切换。默认 100% 也遵循波动规则。
- 命令行用设备事件计时。验证时观察多个波动段和完整忙闲周期，结合驱动实测与 worker 日志。
- `-f` 表示前台运行，Ctrl+C 仅停止本次实例。后台日志在 `scripts/occupy_gpu.log`；启动命令返回后，查看每张卡的日志确认 worker 已运行。原有 `./kill_occupy` 按脚本名停止进程，不区分后端或卡号。
- Python 环境须匹配硬件。MLU 需要 PyTorch 和 `torch_mlu`；NPU 需要 CANN、PyTorch 和 `torch_npu`。探测失败时报告具体错误，不要换后端代跑。

用户要持续监控、查 GPU 进程来源或从浏览器控制 NVIDIA 占卡时，使用 [网页看板](references/dashboard.md)。在目标节点采集，在查看端启动本机网页。默认只读，显式启用控制后，根据用户指定的 GPU 和目标利用率操作。

完整参数和真机验收步骤见 [README.md](README.md)。
