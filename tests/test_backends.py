"""CLI integration tests with fake device runtimes. No vendor SDK or hardware required."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "occupy_gpu.py"

# Expose only the requested backend. A wrong extension, device, dtype or event
# stream fails the worker, just as a missing SDK or wrong current device would.
TORCH = """
import os
import time
backend = os.environ['TEST_BACKEND']
float16, float32 = 'float16', 'float32'
selected = None
clock_ms = 0
def device(name): return name
def randn(n, m, *, device, dtype):
    assert device == ('cuda:0' if backend == 'cuda' else f'{backend}:{selected}')
    assert dtype == (float32 if backend == 'cuda' else float16)
    if backend == 'cuda':
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'
    return device
def empty_like(a): return a
def matmul(a, b, *, out):
    global clock_ms
    assert a == b == out
    clock_ms += 1
    if os.environ.get('TEST_FAIL_CARD') == str(selected):
        raise RuntimeError('simulated operator failure')
class Event:
    def __init__(self, *, enable_timing): assert enable_timing
    def record(self): self.ms = clock_ms
    def elapsed_time(self, end): return end.ms - self.ms
class Runtime:
    Event = Event
    @staticmethod
    def set_device(i):
        global selected
        assert i in range(Runtime.device_count())
        selected = i
    @staticmethod
    def synchronize(): time.sleep(0.0001)
    @staticmethod
    def device_count(): return int(os.environ.get('TEST_COUNT', 2))
if backend == 'cuda':
    cuda = Runtime()
"""


class Backends(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        (self.tmp / "torch.py").write_text(TORCH)
        for backend in ("mlu", "npu"):
            (self.tmp / f"torch_{backend}.py").write_text(
                f"import torch\nassert torch.backend == '{backend}'\ntorch.{backend} = torch.Runtime()\n")
        self.env = dict(os.environ, PYTHONPATH=str(self.tmp),
                        PATH=str(self.tmp) + os.pathsep + os.environ["PATH"])

    def cli(self, backend, *args, **env):
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True,
                              text=True, timeout=10, env=dict(self.env, TEST_BACKEND=backend, **env))

    def exercise(self, backend, spec, pct, cards):
        argv = [sys.executable, str(SCRIPT), spec, "-f", "--pct", str(pct), "--period", "0.05"]
        if backend != "cuda":
            argv.extend(["--backend", backend])
        log = self.tmp / "run.log"
        with log.open("w") as output:
            p = subprocess.Popen(argv, env=dict(self.env, TEST_BACKEND=backend),
                                 stdout=output, stderr=output, start_new_session=True)
            try:
                deadline = time.monotonic() + 8
                label = "GPU" if backend == "cuda" else backend.upper()
                marker = "cycle: busy=10ms" if pct == 20 else "running"
                while time.monotonic() < deadline:
                    data = log.read_text()
                    if all(f"[{label} {card}] {marker}" in data for card in cards):
                        break
                    self.assertIsNone(p.poll(), data)
                    time.sleep(0.02)
                else:
                    self.fail(f"worker(s) did not run:\n{log.read_text()}")
                # Signal the foreground process group, as Ctrl+C from the terminal would.
                os.killpg(p.pid, signal.SIGINT)
                self.assertEqual(p.wait(timeout=5), 0, log.read_text())
            finally:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()

    def test_cuda_default(self):
        self.exercise("cuda", "3", 20, [3])

    def test_mlu_and_npu_all_and_full_load(self):
        for b in ("mlu", "npu"):
            with self.subTest(backend=b):
                self.exercise(b, "all", 20, [0, 1])
                self.exercise(b, "1", 100, [1])

    def test_invalid_devices_fail_before_background_launch(self):
        for b in ("mlu", "npu"):
            for spec, count, message in [("2", "2", "不可见"), ("all", "0", "没有可见")]:
                with self.subTest(backend=b, spec=spec):
                    r = self.cli(b, "--backend", b, spec, TEST_COUNT=count)
                    self.assertNotEqual(r.returncode, 0)
                    self.assertIn(message, r.stderr)
                    self.assertNotIn("正在后台启动", r.stdout)

    def test_missing_extension(self):
        (self.tmp / "torch_npu.py").write_text("raise ImportError('SDK mismatch')\n")
        r = self.cli("npu", "0", "--backend", "npu")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("SDK mismatch", r.stderr)
        self.assertIn("torch_npu", r.stderr)

    def test_worker_failure_releases_other_cards(self):
        r = self.cli("npu", "all", "--backend", "npu", "-f", TEST_FAIL_CARD="1")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("simulated operator failure", r.stderr)

    def test_status_uses_native_monitor_without_torch(self):
        (self.tmp / "torch.py").write_text("raise ImportError('not installed')\n")
        for cmd in ("pgrep", "nvidia-smi", "cnmon", "npu-smi"):
            exe = self.tmp / cmd
            exe.write_text('#!/bin/sh\necho "${0##*/} $*"\n')
            exe.chmod(0o755)
        for b, cmd in [("npu", "npu-smi info"), ("mlu", "cnmon"),
                       ("cuda", "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv")]:
            r = self.cli(b, "--backend", b, "--status")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(cmd, [line.rstrip() for line in r.stdout.splitlines()])
