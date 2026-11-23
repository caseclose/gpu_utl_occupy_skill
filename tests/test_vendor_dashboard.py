"""Exercise the MLU/NPU dashboard agent without loading a real accelerator."""

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from test_backends import TORCH
from test_dashboard import SCRIPTS
from vendor import parse_cnmon, parse_npu_smi

MLU = """| Card VF Name Firmware | Bus-Id | Util Ecc-Error |
| 4 / MLU590 v1 | 0000:03:00.0 | 23% 0 |
| 0 54C 88 W/ 470 W | 1500 MiB/ 81920 MiB | FULL Default |
| 6 / MLU590 v1 | 0000:16:00.0 | 0% 0 |
| 0 38C 65 W/ 470 W | 0 MiB/ 81920 MiB | FULL Default |
| Processes: |
| Card VF PID Command Line MLU Memory Usage |
"""
NPU = """| NPU Name | Health | Power(W) Temp(C) Hugepages-Usage(page) |
| Chip Device | Bus-Id | AICore(%) Memory-Usage(MB) HBM-Usage(MB) |
| 7 Ascend910 | OK | 163.3 36 0 / 0 |
| 0 14 | 0000:81:00.0 | 0 0 / 0 3100 / 65536 |
| 7 Ascend910 | OK | - 36 0 / 0 |
| 1 15 | 0000:83:00.0 | 25 100 / 400 2900 / 65536 |
| NPU Chip | Process id | Process name | Process memory(MB) | Process id in container |
"""
AGENT = SCRIPTS / "dashboard" / "agent.py"


class VendorParsing(unittest.TestCase):
    def test_cnmon_physical_memory_and_process_rows(self):
        cards, procs, complete = parse_cnmon(
            MLU + "| 4 / 321 /usr/bin/python train.py 460 MiB |"
        )
        self.assertTrue(complete)
        self.assertEqual(cards[4, "/"]["uuid"], "MLU-0000:03:00.0")
        self.assertEqual(
            (cards[4, "/"]["util"], cards[4, "/"]["used"], cards[4, "/"]["power"]),
            (23, 1500, 88),
        )
        self.assertEqual(procs[0][:3], ((4, "/"), 321, None))

    def test_npu_multichip_hbm_and_container_pid(self):
        cards, procs, complete = parse_npu_smi(
            NPU + "| 7 1 | 543 | python3 | 256 | 33 |"
        )
        self.assertTrue(complete)
        self.assertEqual(set(cards), {(7, 0), (7, 1)})
        self.assertEqual(cards[7, 1]["uuid"], "NPU-0000:83:00.0-1")
        self.assertEqual((cards[7, 1]["used"], cards[7, 1]["total"]), (2900, 65536))
        self.assertIsNone(cards[7, 1]["power"])
        self.assertEqual(procs[0], ((7, 1), 543, 33, "python3", 256))

    def test_missing_or_unrecognized_process_table_blocks_control(self):
        self.assertFalse(parse_cnmon(MLU + "| permission denied |")[2])
        self.assertFalse(parse_npu_smi(NPU)[2])
        self.assertTrue(
            parse_npu_smi(NPU + "| No running processes found in NPU 7 |")[2]
        )


@unittest.skipUnless(
    sys.platform == "linux", "device collector needs Linux process metadata"
)
class VendorControl(unittest.TestCase):
    def exercise(self, backend):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "state"
            state.mkdir()
            (root / "torch.py").write_text(TORCH + "\n")
            (root / f"torch_{backend}.py").write_text(
                f"import torch\ntorch.{backend}=torch.Runtime()\n"
            )
            if backend == "mlu":
                tool, basic, empty = "cnmon", MLU, "| No running processes found |"
                selected, other = "MLU-0000:16:00.0", "MLU-0000:03:00.0"
                task = lambda pid: f"| 6 / {pid} python3 512 MiB |"
            else:
                tool, basic, empty = (
                    "npu-smi",
                    NPU,
                    "| No running processes found in NPU 7 |",
                )
                selected, other = "NPU-0000:83:00.0-1", "NPU-0000:81:00.0-0"
                task = lambda pid: f"| 7 1 | {pid} | python3 | 512 | {pid} |"
            monitor = root / tool
            monitor.write_text('#!/bin/sh\ncat "$TEST_MONITOR_OUTPUT"\n')
            monitor.chmod(0o700)
            source = root / "monitor.txt"

            def sample(row):
                temp = root / "monitor.tmp"
                temp.write_text(basic + row + "\n")
                temp.replace(source)

            sample(empty)
            mapping = root / "map.json"
            mapping.write_text(json.dumps({selected: 1}))
            env = dict(
                os.environ,
                PYTHONPATH=os.pathsep.join([str(root), str(AGENT.parent)]),
                PATH=str(root) + os.pathsep + os.environ["PATH"],
                TEST_BACKEND=backend,
                TEST_MONITOR_OUTPUT=str(source),
                PYTHONUNBUFFERED="1",
            )
            argv = [
                "collect",
                "--state",
                str(state),
                "--backend",
                backend,
                "--device-map",
                str(mapping),
                "--size",
                "32",
            ]

            def update(card, enabled, revision):
                proc = subprocess.run(
                    [sys.executable, str(AGENT), "update", "--state", str(state)],
                    text=True,
                    input=json.dumps(
                        dict(
                            cards=[card], enabled=enabled, target=20, revision=revision
                        )
                    ),
                    capture_output=True,
                    timeout=5,
                    check=True,
                )
                return json.loads(proc.stdout)

            with (root / "collector.log").open("w+") as log:
                collector = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        "import collector, agent; collector.INTERVAL=.3; agent.main()",
                        *argv,
                    ],
                    env=env,
                    stdout=log,
                    stderr=log,
                )

                def wait(predicate):
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline and collector.poll() is None:
                        try:
                            latest = json.loads((state / "latest.json").read_text())
                        except FileNotFoundError:
                            time.sleep(0.05)
                            continue
                        card = next(g for g in latest["gpus"] if g["uuid"] == selected)
                        if predicate(latest, card):
                            return card
                        time.sleep(0.05)
                    self.fail((root / "collector.log").read_text())

                try:
                    wait(
                        lambda status, card: (
                            status["controllable"] and card["worker_state"] == "off"
                        )
                    )
                    self.assertFalse(update(other, True, 1)["ok"])
                    self.assertTrue(update(selected, True, 1)["ok"])
                    card = wait(lambda status, card: card["worker_state"] == "running")
                    pid = card["worker_pid"]
                    sample(task(pid))
                    wait(
                        lambda status, card: any(
                            p["managed"] for p in status["processes"]
                        )
                    )
                    wait(
                        lambda status, card: bool(
                            re.search(
                                r"\[\w+ 1\] running",
                                (root / "collector.log").read_text(),
                            )
                        )
                    )
                    sample(task(999999))
                    wait(lambda status, card: card["worker_state"] == "yielding")
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
                    sample(empty)
                    worker = wait(
                        lambda status, card: card["worker_state"] == "running"
                    )
                    sample("| process table unavailable |")
                    wait(
                        lambda status, card: (
                            not status["controllable"] and card["worker_state"] == "off"
                        )
                    )
                    with self.assertRaises(ProcessLookupError):
                        os.kill(worker["worker_pid"], 0)
                    sample(empty)
                    wait(lambda status, card: card["worker_state"] == "running")
                    self.assertTrue(update(selected, False, 2)["ok"])
                    wait(lambda status, card: card["worker_state"] == "off")
                finally:
                    collector.send_signal(signal.SIGTERM)
                    collector.wait(timeout=5)

    def test_mlu_and_npu_agent_control(self):
        for backend in ("mlu", "npu"):
            with self.subTest(backend=backend):
                self.exercise(backend)
