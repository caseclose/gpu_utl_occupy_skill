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
from agent import update
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


if __name__ == "__main__":
    unittest.main()
