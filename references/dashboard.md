# 加速卡网页看板

看板可同时显示使用 NVIDIA GPU、寒武纪 MLU 或华为昇腾 NPU 的 Linux 节点。逐卡显示利用率、显存、温度、功耗和最近一小时趋势，并记录进程变动。字段以驱动实际提供的数据为准。采集端需要 Python 3.8+ 和相应监控工具，查看端只需要 Python 3.8+。通过 SSH 读取其他机器时，使用现有 SSH 配置与密钥。

## NVIDIA GPU 节点

在每台要采集的节点上，从仓库根目录启动采集。

```bash
python3 scripts/dashboard/agent.py collect
```

节点默认只做监控。需要网页占卡控制时，可另外编译低优先级 CUDA worker。该 worker 需要支持 Volta 或更新架构的 NVIDIA GPU 和可用的 CUDA 工具链。下例用当前可见 GPU 的架构编译。较旧的工具链可用对应的 `sm_XX` 作为编译目标。 [NVCC 编译选项](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-compiler-driver-nvcc/index.html)

```bash
mkdir -p ~/.local/state/gpu-occupy/dashboard
nvcc -O2 -arch=native scripts/dashboard/pulse.cu \
  -o ~/.local/state/gpu-occupy/dashboard/pulse
python3 scripts/dashboard/agent.py collect \
  --pulse ~/.local/state/gpu-occupy/dashboard/pulse
```

每台节点只需运行一个采集器。主机长期采集时，用该主机已有的服务管理方式运行 `collect`。首次启动时所有 GPU 的占卡开关均关闭。设备设置按 GPU UUID 保存，重启采集器后沿用已保存的设置。最低优先级 CUDA stream 按占空比发短脉冲，整卡利用率以驱动实测为准。

## 寒武纪 MLU 和华为昇腾 NPU 节点

在节点上从仓库根目录启动只读采集。MLU 使用 `cnmon`，NPU 使用 `npu-smi info`。以下两条命令各自对应一种硬件。 [Cambricon 的 cnmon 示例](https://github.com/Cambricon/paddle_mlu/blob/master/install/paddle_install_cn.md) · [昇腾 npu-smi 文档](https://www.hiascend.com/doc_center/source/zh/Atlas%20300I%20Duo%20Inference%20Card/Atlas%20300I%20Duo%20Inference%20Card/apn_016.html)

```bash
python3 scripts/dashboard/agent.py collect --backend mlu --state ~/.local/state/gpu-occupy/mlu
python3 scripts/dashboard/agent.py collect --backend npu --state ~/.local/state/gpu-occupy/npu
```

启用控制时，激活与设备匹配的 PyTorch 环境，并用 `--device-map` 指定监控设备 ID 到 PyTorch **逻辑编号**的 JSON 映射。容器可能重新排列可见设备，先核对对应关系，再写入映射。设备 ID 可从状态目录 `latest.json` 的 `gpus[].uuid` 读取。示例中的地址和编号请替换为当前节点的值。

```json
{"MLU-0000:03:00.0": 0, "MLU-0000:16:00.0": 1}
```

```bash
python3 scripts/dashboard/agent.py collect --backend mlu \
  --state ~/.local/state/gpu-occupy/mlu --device-map /path/to/mlu-map.json
```

NPU 使用同样的方式，将 `--backend` 改为 `npu` 并使用对应的状态目录和映射。NPU ID 示例为 `NPU-0000:87:00.0-1`，末位是芯片 ID。同一节点只映射需要控制的卡。可用 `--size` 调整占卡矩阵边长。MLU／NPU 检测到其他设备进程时会释放自己的 worker。设备进程表无法可靠读取时，控制暂不可用。厂商工具只返回宿主机 PID、采集器又无法匹配容器内 worker 时，优先在能看见对应进程的节点环境运行采集器。

## 在查看端启动网页

在本机创建 JSON 配置。例如，将一个本机采集源和一个 SSH 采集源放在同一页：

```json
{
  "nodes": [
    {"name": "本机", "state": "~/.local/state/gpu-occupy/dashboard"},
    {
      "name": "GPU 节点 B",
      "ssh": "gpu-node-b",
      "agent": "gpu_utl_occupy_skill/scripts/dashboard/agent.py",
      "state": "~/.local/state/gpu-occupy/dashboard"
    }
  ]
}
```

只看 SSH 节点时可以省略本机条目。将 `ssh` 换成可免交互连接的 SSH 别名。将 `agent` 换成远端实际脚本路径，相对路径以 SSH 登录目录为起点。有共享盘时，`ssh` 可以指向跳板机，`state` 指向该节点采集器写入的共享目录。各节点使用独立的状态目录。

```bash
python3 scripts/gpu_dashboard.py --config /path/to/nodes.json
```

打开 <http://127.0.0.1:8801/>。可用 `--port` 改本机端口。用 Ctrl+C 停止查看端。节点上的采集器独立运行，Ctrl+C 停止采集时释放该实例创建的占卡 worker。

展开「集群总览」可比较各节点的在线卡数、利用率、显存、占卡状态和采样时间。节点卡片上的「备注」会设置该卡片的显示名称，备注保存在当前浏览器中。

## 控制和进程来源

选节点后可单点切换一张卡，或按住鼠标拖框选中多张卡。拖框会替换当前节点的选择，按 Esc 可取消。填写 1–100 的目标百分比后，按「开启 / 更新」提交。右上角可选择深色、浅色或跟随系统。

设置通常在下一次采样应用，控制只作用于该采集器创建的 worker。存在其他 GPU 进程且整卡负载达到让出阈值时，NVIDIA 采集器会释放自己的 worker。需要只读采集时，省略 `--pulse` 或 `--device-map`。

目标 T ≤ 20% 时，worker 使用固定占空比。T > 20% 时，每张卡独立选取 T ± 8 个百分点内的占空比，裁剪到 0%–100% 后分段过渡。NVIDIA 的切换相隔 24–40 秒，每次用约 6 秒过渡。现有 NVIDIA 部署更新占卡策略时需重新编译 `pulse` 并重启采集器，已保存的卡设置会继续使用。

归因接受两类证据。能将驱动 PID 与进程及 GPU 设备匹配时，显示相应容器 PID 和命令标记。驱动记录只能与受管进程间接对应时，显示「推定」。显示「未知」表示证据不足。可能相关但未匹配的容器进程列在「容器内疑似任务」。日志保存有限的进程元数据。完整的命令参数和环境变量不会写入日志。

采集间隔为 10 秒。指标和进程变动保存在 `--state` 目录内，保留约 7 天。网页提供最近一小时趋势和每台节点最近 250 条进程变动。
