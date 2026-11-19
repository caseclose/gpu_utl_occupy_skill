# NVIDIA GPU 网页看板

看板支持一台或多台 Linux NVIDIA 节点。逐卡显示利用率、显存、温度、功耗和最近一小时趋势，并记录 GPU 进程的出现、退出和归因变化。采集端需要 Python 3.8+ 和 `nvidia-smi`，查看端只需要 Python 3.8+。通过 SSH 读取其他机器时，使用现有 SSH 配置与密钥。

## 在 GPU 节点上启动采集

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

打开 <http://127.0.0.1:8801/>。可用 `--port` 改本机端口。用 Ctrl+C 停止查看端。节点上的采集器独立运行，Ctrl+C 停止采集时释放该实例创建的 CUDA worker。

## 控制和进程来源

选节点，单选或多选卡，填写 1–100 的目标百分比，然后按「开启 / 更新」或「关闭占卡」。操作只作用于该采集器创建的 worker。页面显示请求值和驱动实测值，设置通常在下一次采样应用。存在其他 GPU 进程且整卡负载达到让出阈值时，采集器释放自己的 worker。无需占卡控制时，用不带 `--pulse` 的方式启动采集。

归因接受两类证据。能将驱动 PID 与进程及 GPU 设备匹配时，显示相应容器 PID 和命令标记。驱动记录只能与受管进程间接对应时，显示「推定」。显示「未知」表示证据不足。可能相关但未匹配的容器进程列在「容器内疑似任务」。日志保存有限的进程元数据。完整的命令参数和环境变量不会写入日志。

采集间隔为 10 秒。指标和进程变动保存在 `--state` 目录内，保留约 7 天。网页提供最近一小时趋势和每台节点最近 250 条进程变动。
