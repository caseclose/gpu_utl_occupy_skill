"""Collect accelerator telemetry and manage only this collector's workers."""
from collections import Counter
import csv
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import pwd
import re
import signal
import subprocess
import time

import vendor

INTERVAL = 10
RETENTION_DAYS = 7
FIELDS = (
    ("index", int), ("name", str), ("uuid", str), ("util", int),
    ("bw", int), ("used", int), ("total", int), ("temp", int), ("power", float),
)
QUERY = "index,name,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw"
DEV = re.compile(r"^/dev/nvidia([0-9]+)$")
SAFE = re.compile(r"^[A-Za-z0-9_.+@-]{1,100}$")
TRAIN = {"torchrun", "deepspeed", "megatron", "accelerate", "train.py", "finetune.py", "sft.py", "pretrain.py"}
INFER = {"vllm", "sglang", "tritonserver", "ollama", "infer.py", "inference.py", "serve.py", "predict.py"}
running = True


def number(raw, convert):
    try:
        return convert(raw)
    except ValueError:
        return None


def query(flag, fields):
    result = subprocess.check_output(
        ["nvidia-smi", flag + "=" + fields, "--format=csv,noheader,nounits"],
        universal_newlines=True, stderr=subprocess.DEVNULL, timeout=6)
    return [tuple(x.strip() for x in row) for row in csv.reader(result.splitlines()) if row]


def read_gpus():
    rows = query("--query-gpu", QUERY)
    cards = [dict((key, number(value, convert)) for (key, convert), value in zip(FIELDS, row))
             for row in rows]
    if (not cards or any(not isinstance(g.get("index"), int) or not g.get("uuid") for g in cards)
            or len({g["uuid"] for g in cards}) != len(cards)):
        raise ValueError("nvidia-smi did not report distinct GPUs")
    return sorted(cards, key=lambda g: g["index"])


def badged_commands(args, exe):
    # Inspect only a short prefix and retain matched, fixed markers. Do not persist argv.
    for arg in [exe] + args[:8]:
        token = arg.rsplit("/", 1)[-1].lower()
        if token == "occupy_gpu.py":
            return "占卡", token
        if token in TRAIN or re.fullmatch(r"(train|finetune|pretrain|sft|grpo|ppo)(_[a-z0-9_]+)?\.py", token):
            return "训练", token
        if token in INFER or re.fullmatch(r"(vllm|sglang)(\.[a-z0-9_.]+)?|(infer|inference|predict)(_[a-z0-9_]+)?\.py", token):
            return "推理", token
    return "未知", None


def process_info(path):
    """Read visible PID metadata and device FDs without retaining command arguments."""
    try:
        rows = (path / "status").read_text(errors="replace").splitlines()
        fields = dict(row.split(":", 1) for row in rows if ":" in row)
        local_pid = int(path.name)
        uid = int(fields["Uid"].split()[0])
        name = fields["Name"].strip()
        exe = os.path.basename(os.readlink(str(path / "exe")))
        devices = set()
        for entry in (path / "fd").iterdir():
            try:
                match = DEV.fullmatch(os.readlink(str(entry)))
            except OSError:
                continue
            if match:
                devices.add(int(match.group(1)))
        if not devices:
            return None
        with (path / "cmdline").open("rb") as stream:
            args = stream.read(8192).decode("utf-8", "replace").split("\x00")
        category, marker = badged_commands(args, exe)
        try:
            owner = pwd.getpwuid(uid).pw_name
        except KeyError:
            owner = str(uid)
        start_ticks = (path / "stat").read_text().rsplit(")", 1)[1].split()[19]
        ids = [int(pid) for pid in fields.get("NSpid", str(local_pid)).split()]
        return dict(local_pid=local_pid, ppid=int(fields["PPid"]), user=owner, start_ticks=start_ticks,
                    command=exe if SAFE.fullmatch(exe) else name if SAFE.fullmatch(name) else None,
                    category=category, marker=marker, ids=ids, devices=devices)
    except (OSError, ValueError, KeyError, IndexError):
        return None


def process_rows(cards, workers):
    uuid_index = {g["uuid"]: g["index"] for g in cards}
    index_uuid = {index: uuid for uuid, index in uuid_index.items()}
    rows = query("--query-compute-apps", "gpu_uuid,pid,used_gpu_memory,process_name")
    devices = {}
    for uuid, pid, mem, name in rows:
        if uuid not in uuid_index or not pid.isdigit():
            continue
        identity = name.strip()
        if identity.startswith("[") or identity == "N/A":
            identity = ""
        command = os.path.basename(identity)
        devices.setdefault(uuid_index[uuid], []).append(dict(
            gpu=uuid_index[uuid], pid=int(pid), memory=number(mem, int),
            command=command if SAFE.fullmatch(command) else None,
        ))
    try:
        pmon = subprocess.check_output(["nvidia-smi", "pmon", "-s", "m", "-c", "1"],
                                       universal_newlines=True, stderr=subprocess.DEVNULL, timeout=4)
        for line in pmon.splitlines():
            parts = line.split()
            if (len(parts) >= 6 and parts[0].isdigit()
                    and parts[1].isdigit() and "G" in parts[2]):
                gpu, pid = int(parts[0]), int(parts[1])
                if gpu not in uuid_index.values():
                    continue
                items = devices.setdefault(gpu, [])
                if not any(p["pid"] == pid for p in items):
                    command = parts[-1]
                    items.append(dict(gpu=gpu, pid=pid, memory=number(parts[3], int),
                                      command=command if SAFE.fullmatch(command) else None))
    except (OSError, subprocess.SubprocessError):
        pass
    visible = [info for entry in Path("/proc").glob("[0-9]*")
               if entry.name.isdigit() for info in [process_info(entry)] if info]
    indices = {}
    # NVIDIA's CUDA index and /dev/nvidia minor are independent on these hosts.
    for file in Path("/proc/driver/nvidia/gpus").glob("*/information"):
        try:
            keys = dict((k.strip(), v.strip()) for k, v in (
                line.split(":", 1) for line in file.read_text().splitlines() if ":" in line))
            uuid = keys.get("GPU UUID")
            if uuid in uuid_index:
                indices[int(keys["Device Minor"])] = uuid_index[uuid]
        except (OSError, ValueError, KeyError):
            continue
    for process in visible:
        process["devices"] = {indices[d] for d in process["devices"] if d in indices}
    local = {p["local_pid"]: p for p in visible}
    result = []
    known = {p.pid for p, _ in workers.values()}
    for index, entries in devices.items():
        worker = workers.get(index_uuid[index])
        for row in entries:
            matches = [p for p in visible if row["pid"] in p["ids"] and index in p["devices"]]
            verified = len(matches) == 1
            if not verified and worker and len(entries) == 1 and row["command"] in (None, "pulse"):
                candidate = local.get(worker[0].pid)
                # The management PID has a verified CUDA device FD. NVML may expose only the host PID.
                if candidate and index in candidate["devices"] and not any(
                    index in p["devices"] and p["local_pid"] not in known for p in visible
                ):
                    matches = [candidate]
            if len(matches) == 1:
                p = matches[0]
                guard = p["local_pid"] in known
                row.update(local_pid=p["local_pid"], ppid=p["ppid"], user=p["user"], start_ticks=p["start_ticks"],
                           command=p["command"], category="占卡" if guard else p["category"],
                           confidence="已匹配" if verified else "推定", managed=guard,
                           evidence=("受管 CUDA 进程与设备匹配" if verified else "单条驱动记录与受管 CUDA 进程的设备相符")
                           if guard else ("匹配设备与进程，命令标记 " + p["marker"] if p["marker"] else "匹配进程，任务目的无法判定"))
            else:
                category, marker = badged_commands([], row["command"] or "")
                row.update(local_pid=None, category=category, managed=False, confidence="推定" if marker else "未知",
                           evidence=("驱动上报名称 " + marker) if marker else "驱动 PID 无法匹配到容器内进程")
            result.append(row)
    candidates = [dict(gpus=sorted(p["devices"]), local_pid=p["local_pid"], command=p["command"],
                       user=p["user"], category=p["category"], evidence="容器内设备 FD 与命令标记 " + p["marker"])
                  for p in visible if p["marker"] and p["local_pid"] not in
                  {entry.get("local_pid") for entry in result}]
    return sorted(result, key=lambda p: (p["gpu"], p["pid"])), candidates


def stop(worker):
    if worker.poll() is None:
        worker.terminate()
        try:
            worker.wait(timeout=2)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()


def publish(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def append(directory, prefix, value, now):
    day = datetime.utcfromtimestamp(now).strftime("%Y-%m-%d")
    with (directory / (prefix + "-" + day + ".ndjson")).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def mark_stop(signum, frame):
    global running
    running = False


def vendor_process_rows(cards, reported, workers):
    own = {}
    for uuid, (worker, _) in workers.items():
        if worker.poll() is not None:
            continue
        try:
            lines = Path(f"/proc/{worker.pid}/status").read_text().splitlines()
            ids = next((line.split()[1:] for line in lines if line.startswith("NSpid:")), [])
        except OSError:
            continue
        own[uuid] = (worker.pid, {worker.pid, *map(int, ids)})
    result = []
    for key, pid, container_pid, raw_command, memory in reported:
        if key not in cards:
            continue
        card = cards[key]
        local, ids = own.get(card["uuid"], (None, set()))
        managed = container_pid == local if container_pid is not None else pid in ids
        command = raw_command.split(maxsplit=1)[0].rsplit("/", 1)[-1] if raw_command else ""
        command = command if SAFE.fullmatch(command) else None
        category, marker = badged_commands([], command or "")
        result.append(dict(gpu=card["index"], pid=pid, local_pid=local if managed else container_pid,
                           label=card["label"], command=command, memory=memory, managed=managed,
                           category="占卡" if managed else category,
                           confidence="已匹配" if managed else "推定" if marker else "未知",
                           evidence="厂商进程记录匹配受管进程" if managed else "厂商工具上报的进程记录"))
    return result


def reconcile_workers(gpus, processes, config, pulse, workers, vendor_run=None):
    vendor_mode = vendor_run is not None
    index_uuid = {gpu["index"]: gpu["uuid"] for gpu in gpus}
    for uuid in workers.keys() - index_uuid.values():
        stop(workers.pop(uuid)[0])
    counts = Counter(p["gpu"] for p in processes)
    external = {p["gpu"] for p in processes if not p["managed"]
                and (vendor_mode or index_uuid[p["gpu"]] not in workers or counts[p["gpu"]] > 1
                     or p["confidence"] == "已匹配")}
    for card in gpus:
        index = card["index"]
        uuid = card["uuid"]
        desired = config["cards"].get(uuid, {"enabled": False, "target": 20})
        child = workers.get(uuid)
        threshold = min(100, desired["target"] * 2 + 4) if child else desired["target"]
        paused = desired["enabled"] and index in external and (vendor_mode or (card["util"] or 0) >= threshold)
        controllable = card["controllable"] if vendor_mode else bool(pulse)
        want = controllable and desired["enabled"] and not paused
        if child and (not want or child[1] != desired["target"] or child[0].poll() is not None):
            stop(workers.pop(uuid)[0])
            child = None
        if want and child is None:
            worker = vendor_run(card, desired["target"]) if vendor_mode else subprocess.Popen(
                [str(pulse), str(desired["target"])],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=card["uuid"]),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            child = (worker, desired["target"])
            workers[uuid] = child
        card.update(requested=desired, worker_pid=child[0].pid if child else None,
                    worker_state="running" if child else "yielding" if paused else "off")


def supervise(directory, pulse=None, backend="cuda", device_map=None, size=2048):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    status_path = directory / "latest.json"
    config_path = directory / "config.json"
    with (directory / "collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (directory / "config.lock").open("a") as config_lock:
            fcntl.flock(config_lock, fcntl.LOCK_EX)
            if not config_path.exists():
                publish(config_path, dict(revision=1, cards={}))
        signal.signal(signal.SIGTERM, mark_stop)
        signal.signal(signal.SIGINT, mark_stop)
        workers, active = {}, {}
        try:
            active = {p["key"]: p for p in json.loads(status_path.read_text(encoding="utf-8")).get("processes", [])}
        except (OSError, KeyError, ValueError):
            pass
        cleanup_at = 0

        def vendor_run(card, target):
            return vendor.launch(backend, device_map[card["uuid"]], target, size)

        try:
            while running:
                started = time.monotonic()
                now = int(time.time())
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    if backend == "cuda":
                        gpus = read_gpus()
                        processes, candidates = process_rows(gpus, workers)
                    else:
                        cards, reported, complete = vendor.sample(backend)
                        gpus = sorted(cards.values(), key=lambda card: card["index"])
                        for card in gpus:
                            card["controllable"] = complete and card["util"] is not None and card["uuid"] in (device_map or {})
                        processes, candidates = vendor_process_rows(cards, reported, workers), []
                    reconcile_workers(gpus, processes, config, pulse, workers,
                                      vendor_run if backend != "cuda" else None)
                    current = {}
                    for p in processes:
                        # GPU index and observed PID allow consistency even when namespace metadata becomes available later.
                        key = "{}:{}".format(p["gpu"], p["pid"])
                        previous = active.get(key)
                        same = previous and not (previous.get("start_ticks") and p.get("start_ticks")
                                                 and previous["start_ticks"] != p["start_ticks"])
                        if previous and not same:
                            append(directory, "events", dict(time=now, action="离开", **previous), now)
                        p.update(key=key, first_seen=previous["first_seen"] if same else now, last_seen=now)
                        if not same:
                            append(directory, "events", dict(time=now, action="出现", **p), now)
                        elif previous["category"] != p["category"]:
                            append(directory, "events", dict(time=now, action="更新归因", **p), now)
                        current[key] = p
                    for key in active.keys() - current.keys():
                        append(directory, "events", dict(time=now, action="离开", **active[key]), now)
                    active = current
                    control_available = bool(pulse) if backend == "cuda" else any(g["controllable"] for g in gpus)
                    publish(status_path, dict(time=now, backend=backend, controllable=control_available, gpus=gpus,
                                              processes=processes, candidates=candidates, revision=config["revision"]))
                    append(directory, "metrics", dict(t=now, g={
                        (g["index"] if backend == "cuda" else g["uuid"]):
                        [g[k] for k in ("util", "used", "total", "temp", "power")] for g in gpus
                    }), now)
                    if now - cleanup_at >= 3600:
                        cutoff = datetime.utcfromtimestamp(now - RETENTION_DAYS * 86400).strftime("%Y-%m-%d")
                        for path in directory.glob("*.ndjson"):
                            if path.name.startswith(("metrics-", "events-")) and path.stem.split("-", 1)[1] < cutoff:
                                path.unlink()
                        cleanup_at = now
                except (OSError, ValueError, KeyError, AssertionError, subprocess.SubprocessError) as exc:
                    if backend != "cuda":
                        for worker, _ in workers.values():
                            stop(worker)
                        workers.clear()
                    publish(directory / "health.json", dict(time=now, error=type(exc).__name__))
                    print("sample failed: {}".format(type(exc).__name__), flush=True)
                deadline = started + INTERVAL
                while running and time.monotonic() < deadline:
                    time.sleep(0.2)
        finally:
            for worker, _ in workers.values():
                stop(worker)
