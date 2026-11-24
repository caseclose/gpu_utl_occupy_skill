"""Exercise shared display preferences with independent local HTTP clients."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from gpu_dashboard import Backend, DashboardServer, Node
from view_preferences import ViewPreferences


class SharedPreferencesTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "view-preferences.json"
        self.names = ["Node A", "Node B", "Node C"]
        backend = Backend([Node(name, directory.name) for name in self.names])
        self.store = ViewPreferences(self.path, self.names)
        self.server = DashboardServer(0, backend, self.store)
        self.addCleanup(self.server.server_close)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.shutdown)
        self.origin = self.server.origin

    def read(self):
        with urlopen(self.origin + "/api/status", timeout=3) as response:
            return json.load(response)["view_preferences"]

    def save(self, value, origin=None):
        request = Request(
            self.origin + "/api/view-preferences",
            data=json.dumps(value).encode(),
            headers={
                "Content-Type": "application/json",
                "Origin": origin or self.origin,
            },
        )
        with urlopen(request, timeout=3) as response:
            return json.load(response)["view_preferences"]

    def test_note_and_order_updates_share_state_without_losing_other_fields(self):
        a, b, c = self.names
        with ThreadPoolExecutor() as pool:
            note = pool.submit(self.save, {"node": a, "note": "shared note"})
            move = pool.submit(self.save, {"node": c, "destination": a})
            note.result()
            move.result()
        expected = {"revision": 2, "order": [c, a, b], "notes": {a: "shared note"}}
        self.assertEqual(self.read(), expected)
        self.assertEqual(ViewPreferences(self.path, self.names).read(), expected)
        self.save({"node": a, "note": ""})
        self.assertEqual(self.read()["notes"], {})
        self.assertEqual(self.read()["order"], [c, a, b])

    def test_legacy_import_is_applied_once_and_ignores_unknown_nodes(self):
        a, b, _ = self.names
        self.assertEqual(self.read()["revision"], 0)
        self.save({"import": {"order": [], "notes": {}}})
        self.assertEqual(self.read()["revision"], 0)
        saved = self.save(
            {
                "import": {
                    "order": [b, "retired", a, b],
                    "notes": {b: "legacy", "retired": "old"},
                }
            }
        )
        self.assertEqual(
            saved, {"revision": 1, "order": [b, a], "notes": {b: "legacy"}}
        )
        self.save({"import": {"order": [a, b], "notes": {a: "other browser"}}})
        self.assertEqual(self.read(), saved)

    def test_invalid_and_cross_origin_requests_do_not_write(self):
        a, b, _ = self.names
        for body, origin, status in [
            ({"node": "unknown", "note": "x"}, self.origin, 400),
            ({"node": a, "destination": a}, self.origin, 400),
            ({"node": a, "note": "x" * 41}, self.origin, 400),
            ({"node": b, "note": "x"}, "http://example.invalid", 403),
        ]:
            with self.assertRaises(HTTPError) as result:
                self.save(body, origin)
            self.assertEqual(result.exception.code, status)
        self.assertEqual(self.read()["revision"], 0)
        self.assertFalse(self.path.exists())
