"""MLU and NPU telemetry and single-device workers for the web collector."""

import ctypes
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

COMMANDS = {"mlu": ["cnmon"], "npu": ["npu-smi", "info"]}
PCI = re.compile(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", re.I)
MLU_DEVICE = re.compile(r"(\d+)\s+(\S+)\s+(MLU[\w.-]+)(?:\s+.*)?", re.I)
MLU_MEMORY = re.compile(r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB", re.I)
NPU_MEMORY = re.compile(r"(\d+)\s*/\s*(\d+)")
MLU_PROCESS = re.compile(r"(\d+)\s+(\S+)\s+(\d+)\s+(.*?)\s+([\d.]+)\s*MiB", re.I)


def parse_number(raw, convert=int):
    try:
        return convert(raw)
    except (TypeError, ValueError):
        return None


def accelerator(label, name, identifier, util, memory, temp, power):
    return dict(
        label=label,
        name=name,
        uuid=identifier,
        util=util,
        bw=None,
        used=memory[0],
        total=memory[1],
        temp=temp,
        power=power,
    )


def parts_of(output):
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("|") and line.endswith("|"):
            yield [part.strip() for part in line[1:-1].split("|")]


def parse_cnmon(output):
    cards, processes = {}, []
    current, process_table, complete, evidence = None, False, True, False
    for parts in parts_of(output):
        first = parts[0]
        if first.lower().startswith("processes:"):
            process_table = True
            continue
        if process_table:
            if first.startswith("Card ") and "PID" in first:
                continue
            if re.fullmatch(r"No running processes found", first, re.I):
                evidence = True
                continue
            row = MLU_PROCESS.fullmatch(first)
            if not row:
                complete = False
                continue
            card, partition, pid, command, memory = row.groups()
            processes.append(
                ((int(card), partition), int(pid), None, command, int(float(memory)))
            )
            evidence = True
            continue
        device = MLU_DEVICE.fullmatch(first)
        if device and len(parts) >= 3:
            card, partition, name = device.groups()
            current = (int(card), partition)
            if current in cards:
                raise ValueError("duplicate CNMON device")
            bus = parts[1].lower()
            identity = bus if PCI.fullmatch(bus) else card
            util = re.match(r"(\d+)\s*%", parts[2])
            label = f"MLU {card}" if partition == "/" else f"MLU {card}/{partition}"
            identifier = (
                f"MLU-{identity}" if partition == "/" else f"MLU-{identity}-{partition}"
            )
            cards[current] = accelerator(
                label,
                name,
                identifier,
                int(util[1]) if util else None,
                (None, None),
                None,
                None,
            )
            continue
        if current is not None and len(parts) >= 2:
            memory = MLU_MEMORY.search(parts[1])
            if memory:
                temp = re.search(r"(-?\d+)\s*C\b", first)
                power = re.search(r"([\d.]+)\s*W\b", first)
                cards[current].update(
                    used=int(memory[1]),
                    total=int(memory[2]),
                    temp=int(temp[1]) if temp else None,
                    power=float(power[1]) if power else None,
                )
                current = None
    return cards, processes, complete and evidence


def parse_npu_smi(output):
    cards, processes, covered = {}, [], set()
    current, process_table, complete = None, False, True
    for parts in parts_of(output):
        first = parts[0]
        if (
            first.startswith("NPU")
            and len(parts) > 1
            and parts[1].lower().startswith("process id")
        ):
            process_table = True
            continue
        if process_table:
            empty = re.fullmatch(
                r"No running processes found in NPU\s+(\d+)", first, re.I
            )
            if empty:
                covered.add(int(empty[1]))
                continue
            address = first.split()
            pid = parse_number(parts[1]) if len(parts) >= 4 else None
            if (
                len(address) != 2
                or not all(value.isdecimal() for value in address)
                or pid is None
            ):
                complete = False
                continue
            key = tuple(map(int, address))
            container_pid = parse_number(parts[4]) if len(parts) > 4 else None
            processes.append(
                (key, pid, container_pid, parts[2], parse_number(parts[3]))
            )
            covered.add(key[0])
            continue
        address = first.split(maxsplit=1)
        if len(parts) < 3 or not address or not address[0].isdecimal():
            continue
        bus = parts[1].lower()
        if PCI.fullmatch(bus) and current is not None:
            npu, name, temp, power = current
            chip = int(address[0])
            key = (npu, chip)
            if key in cards:
                raise ValueError("duplicate npu-smi device")
            memory = next(
                (
                    (int(used), int(total))
                    for used, total in reversed(NPU_MEMORY.findall(parts[2]))
                    if int(total)
                ),
                (None, None),
            )
            fields = parts[2].split()
            cards[key] = accelerator(
                f"NPU {npu}/{chip}",
                name,
                f"NPU-{bus}-{chip}",
                parse_number(fields[0]) if fields else None,
                memory,
                temp,
                power,
            )
            current = None
        elif len(address) == 2:
            fields = parts[2].split()
            current = (
                int(address[0]),
                address[1],
                parse_number(fields[1]) if len(fields) > 1 else None,
                parse_number(fields[0], float) if fields else None,
            )
    return cards, processes, complete and all(npu in covered for npu, _ in cards)


def sample(backend):
    output = subprocess.check_output(
        COMMANDS[backend],
        text=True,
        timeout=6,
        stderr=subprocess.DEVNULL,
        env=dict(os.environ, LC_ALL="C"),
    )
    parser = parse_cnmon if backend == "mlu" else parse_npu_smi
    cards, processes, complete = parser(output)
    if not cards or len({x["uuid"] for x in cards.values()}) != len(cards):
        raise ValueError(f"{backend} monitor did not report distinct devices")
    for index, key in enumerate(sorted(cards)):
        cards[key]["index"] = index
    complete = complete and all(key in cards for key, *_ in processes)
    return cards, processes, complete


def launch(backend, logical_index, target, size):
    return subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            backend,
            str(logical_index),
            str(target),
            str(size),
            str(os.getpid()),
        ]
    )


def load_device_map(path, backend):
    # The administrator supplies a stable monitor ID -> runtime index mapping.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from occupy_gpu import visible_device_count

    mapping = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(mapping, dict)
        or not mapping
        or not all(
            isinstance(key, str)
            and key.startswith(backend.upper() + "-")
            and type(value) is int
            for key, value in mapping.items()
        )
    ):
        raise ValueError("设备映射必须是监控 ID 到 PyTorch 逻辑编号的非空 JSON 对象")
    count = visible_device_count(backend)
    if len(set(mapping.values())) != len(mapping) or any(
        not 0 <= v < count for v in mapping.values()
    ):
        raise ValueError("设备映射的 PyTorch 逻辑编号必须唯一且当前可见")
    return mapping


if __name__ == "__main__":
    backend, index, target, size, parent = sys.argv[1:]
    parent = int(parent)
    if (
        os.getppid() != parent
        or ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0
        or os.getppid() != parent
    ):
        sys.exit("collector exited before worker startup")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from occupy_gpu import worker

    worker(int(index), int(size), float(target), 10.0, backend)
