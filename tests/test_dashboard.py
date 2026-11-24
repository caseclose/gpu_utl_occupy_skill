"""Dashboard configuration and worker isolation, using an isolated local fixture."""
from contextlib import ExitStack
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path[:0] = [str(SCRIPTS / "dashboard")]
from agent import main, update
from collector import reconcile_workers


class DashboardAgentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.state = Path(temp.name)
        self.config = {"revision": 1, "cards": {}}
        self.gpu_a = "GPU-a"
        self.gpu_b = "GPU-b"
        self.save("config.json", self.config)
        self.save("latest.json", {"controllable": True, "time": time.time(),
                                  "gpus": [{"uuid": self.gpu_a}, {"uuid": self.gpu_b}]})

    def save(self, file, value):
        (self.state / file).write_text(json.dumps(value))

    def request(self, cards, **overrides):
        payload = dict(cards=cards, enabled=True, target=20, revision=1)
        return update(self.state, dict(payload, **overrides))

    def test_controls_are_uuid_scoped_with_revision(self):
        accepted = self.request([self.gpu_b], target=35)
        self.assertTrue(accepted["ok"])
        self.assertEqual(accepted["config"]["cards"], {self.gpu_b: {"enabled": True, "target": 35}})
        self.assertEqual(accepted["config"]["card_revisions"], {self.gpu_b: 2})
        self.assertEqual(self.request([self.gpu_a])["status"], 409)
        with self.assertRaises(ValueError):
            self.request(["GPU-unlisted"], revision=2)
        disabled = self.request([self.gpu_b], enabled=False, revision=2)
        self.assertFalse(disabled["config"]["cards"][self.gpu_b]["enabled"])

    def test_read_only_and_stale_status_reject_updates(self):
        for status in (dict(controllable=False, time=time.time()), dict(controllable=True, time=time.time()-60)):
            self.save("latest.json", status)
            self.assertEqual(self.request([self.gpu_b])["status"], 409)
        self.assertEqual(json.loads((self.state / "config.json").read_text()), self.config)

    def test_only_selected_worker_is_started_changed_and_stopped(self):
        cards = [{"index": 2, "uuid": self.gpu_a, "util": 0},
                 {"index": 5, "uuid": self.gpu_b, "util": 0}]
        running = {}
        settings = {self.gpu_b: {"enabled": True, "target": 20}}
        with ExitStack() as stack:
            worker = lambda pid: Mock(spec=subprocess.Popen, poll=Mock(return_value=None), pid=pid)
            first, second = worker(123), worker(124)
            spawn = stack.enter_context(patch("collector.subprocess.Popen", side_effect=[first, second]))
            reconcile_workers(cards, [], {"cards": settings}, Path("/example/pulse"), running)
            self.assertEqual(set(running), {self.gpu_b})
            self.assertEqual(spawn.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], self.gpu_b)
            self.assertEqual(spawn.call_args.args[0][-1], "20")
            cards[1]["index"] = 7
            settings[self.gpu_b]["target"] = 30
            reconcile_workers(cards, [], {"cards": settings}, Path("/example/pulse"), running)
            first.terminate.assert_called_once()
            self.assertEqual(spawn.call_args.args[0][-1], "30")
            settings[self.gpu_b]["enabled"] = False
            reconcile_workers(cards, [], {"cards": settings}, Path("/example/pulse"), running)
            second.terminate.assert_called_once()
            self.assertFalse(running)
            self.assertEqual(spawn.call_count, 2)

    def test_recovery_holds_saved_load_until_a_new_request_for_each_card(self):
        cases = (("cuda", None), ("mlu", Mock()), ("npu", Mock()))
        for backend, vendor_run in cases:
            with self.subTest(backend=backend):
                self.save("config.json", {"revision": 4, "cards": {
                    self.gpu_a: {"enabled": True, "target": 60},
                    self.gpu_b: {"enabled": True, "target": 75},
                }})
                cards = [{"index": i, "uuid": uuid, "util": 0, "controllable": True}
                         for i, uuid in enumerate((self.gpu_a, self.gpu_b))]
                workers = {}
                with patch("collector.subprocess.Popen") as spawn:
                    config = json.loads((self.state / "config.json").read_text())
                    reconcile_workers(cards, [], config, "/example/pulse", workers,
                                      vendor_run, recovery_revision=4)
                    self.assertEqual([c["worker_state"] for c in cards], ["recovery_hold"] * 2)
                    self.assertFalse(workers)
                    updated = self.request([self.gpu_b], revision=4, target=80)
                    self.assertTrue(updated["ok"])
                    reconcile_workers(cards, [], updated["config"], "/example/pulse", workers,
                                      vendor_run, recovery_revision=4)
                    self.assertEqual([c["worker_state"] for c in cards], ["recovery_hold", "running"])
                    self.assertEqual(set(workers), {self.gpu_b})
                    if vendor_run is None:
                        spawn.assert_called_once()
                        self.assertEqual(spawn.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], self.gpu_b)
                    else:
                        spawn.assert_not_called()
                        vendor_run.assert_called_once_with(cards[1], 80)

    def test_recovery_keeps_telemetry_when_control_dependencies_are_unavailable(self):
        args = ["agent.py", "collect", "--state", str(self.state), "--backend", "npu",
                "--device-map", str(self.state / "missing.json"), "--recover-monitor"]
        with patch("sys.argv", args), patch("agent.supervise") as start, patch("sys.stderr"):
            main()
        start.assert_called_once_with(self.state.resolve(), None, "npu", None, 2048,
                                      recover_monitor=True)


if __name__ == "__main__":
    unittest.main()
