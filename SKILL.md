---
name: gpu-occupy
description: 在指定 NVIDIA GPU、寒武纪 MLU 或华为昇腾 NPU 上产生矩阵运算负载，支持多卡和目标平均利用率。用户要求占卡、设置设备负载或查看占用状态时使用。
---

# 加速卡占用

在本 Skill 的 `scripts/` 目录运行命令。根据用户指定的硬件选择后端：

| 硬件 | 查看状态 | 示例：在卡 0 上前台运行，目标平均 20% |
|---|---|---|
| NVIDIA GPU | `./occupy_gpu --status` | `./occupy_gpu 0 --pct 20 -f` |
| 寒武纪 MLU | `./occupy_gpu --backend mlu --status` | `./occupy_gpu --backend mlu 0 --pct 20 -f` |
| 华为昇腾 NPU | `./occupy_gpu --backend npu --status` | `./occupy_gpu --backend npu 0 --pct 20 -f` |

- MLU / NPU 的编号是当前 PyTorch 环境可见的逻辑编号；管理工具可能显示宿主机物理编号。`0,1` 选两卡，`all` 选该后端的全部可见卡，不检查卡是否空闲。先查看状态，不要把请求的单卡扩大成全部卡。
- 省略 `--pct` 即满载。限流时用设备事件计时；用多个完整周期的采样均值验证，单个瞬时读数不代表长期均值。
- `-f` 表示前台运行，Ctrl+C 仅停止本次实例。后台日志在 `scripts/occupy_gpu.log`；启动命令返回后，查看每张卡的日志确认 worker 已运行。原有 `./kill_occupy` 按脚本名停止进程，不区分后端或卡号。
- Python 环境须匹配硬件。MLU 需要 PyTorch 和 `torch_mlu`；NPU 需要 CANN、PyTorch 和 `torch_npu`。探测失败时报告具体错误，不要换后端代跑。

完整参数和真机验收步骤见 [README.md](README.md)。
