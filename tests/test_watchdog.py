"""Run the Linux watchdog against fake NVIDIA telemetry without a GPU."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from test_dashboard import SCRIPTS


DASHBOARD = SCRIPTS / "dashboard"


@unittest.skipUnless(
    sys.platform == "linux" and all(shutil.which(x) for x in ("bash", "flock", "setsid")),
    "watchdog requires Linux and util-linux",
)
class WatchdogTests(unittest.TestCase):
    def test_restarts_an_exited_collector_without_replaying_occupancy(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            state = folder / "node state"
            state.mkdir()
            config = {"revision": 3, "cards": {"GPU-a": {"enabled": True, "target": 80}}}
            config_path = state / "config.json"
            config_path.write_text(json.dumps(config))
            monitor = folder / "nvidia-smi"
            monitor.write_text(
                '#!/bin/sh\ncase "$1" in\n'
                '  --query-gpu=*) echo "0,Test card,GPU-a,0,0,0,8192,35,25" ;;\n'
                'esac\n'
            )
            monitor.chmod(0o700)
            pulse = folder / "pulse"
            pulse.write_text('#!/bin/sh\ntouch "$UNEXPECTED_OCCUPANCY"\n')
            pulse.chmod(0o700)
            marker = folder / "unexpected-load"
            env = dict(os.environ, PATH=f"{folder}{os.pathsep}{os.environ['PATH']}",
                       GPU_OCCUPY_PYTHON=sys.executable, UNEXPECTED_OCCUPANCY=str(marker))
            latest_path = state / "latest.json"

            def wait_for(check):
                deadline = time.monotonic() + 12
                while time.monotonic() < deadline:
                    value = check()
                    if value:
                        return value
                    time.sleep(0.05)
                diagnostics = [
                    path.read_text()
                    for path in (folder / "watchdog.log", state / "collector.log")
                    if path.exists()
                ]
                self.fail("watchdog verification timed out\n" + "\n".join(diagnostics))

            def latest_if(predicate):
                try:
                    latest = json.loads(latest_path.read_text())
                except FileNotFoundError:
                    return None
                return latest if predicate(latest) else None

            def waiting_for_tick(pid):
                children = Path(f"/proc/{pid}/task/{pid}/children")
                for child in children.read_text().split():
                    try:
                        args = Path(f"/proc/{child}/cmdline").read_bytes().split(b"\0")
                    except FileNotFoundError:
                        continue
                    if args[:2] == [b"sleep", b"900"]:
                        return True
                return False

            original = subprocess.Popen(
                [sys.executable, str(DASHBOARD / "agent.py"), "collect", "--state", str(state)],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            watchdog = None
            log_path = folder / "watchdog.log"
            try:
                wait_for(lambda: latest_if(lambda status: status["pid"] == original.pid))
                with log_path.open("w") as log:
                    watchdog = subprocess.Popen(
                        [str(DASHBOARD / "watchdog.sh"), str(state), "--pulse", str(pulse)],
                        env=env, stdout=log, stderr=log,
                    )
                wait_for(lambda: waiting_for_tick(watchdog.pid))
                self.assertEqual(json.loads(latest_path.read_text())["pid"], original.pid)
                self.assertNotIn("restart requested", log_path.read_text())
                original.kill()
                original.wait(timeout=5)
                watchdog.send_signal(signal.SIGUSR1)
                recovered = wait_for(lambda: latest_if(lambda status: status["pid"] != original.pid))
                self.assertTrue(recovered["controllable"])
                self.assertEqual(recovered["gpus"][0]["worker_state"], "recovery_hold")
                self.assertEqual(recovered["gpus"][0]["requested"]["target"], 80)
                self.assertIsNone(recovered["gpus"][0]["worker_pid"])
                self.assertEqual(json.loads(config_path.read_text()), config)
                self.assertFalse(marker.exists())
                wait_for(lambda: waiting_for_tick(watchdog.pid))
            finally:
                if watchdog is not None:
                    watchdog.terminate()
                    watchdog.wait(timeout=5)
                if original.poll() is None:
                    original.terminate()
                    original.wait(timeout=5)
                if log_path.exists():
                    lines = log_path.read_text().splitlines()
                    for line in lines:
                        if "collector restart requested (pid " in line:
                            pid = int(line.rsplit(" ", 1)[1].strip().rstrip(")"))
                            cmdline = Path(f"/proc/{pid}/cmdline")
                            if cmdline.exists() and os.fsencode(str(state)) in cmdline.read_bytes().split(b"\0"):
                                try:
                                    os.kill(pid, signal.SIGTERM)
                                except ProcessLookupError:
                                    pass
