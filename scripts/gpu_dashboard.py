"""Serve a local NVIDIA dashboard for one or more configured telemetry sources."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

ASSETS = Path(__file__).resolve().parent / "dashboard"
ROUTES = {
    "/": ("page.html", "text/html"),
    "/app.js": ("app.js", "application/javascript"),
    "/style.css": ("style.css", "text/css"),
}


@dataclass
class Node:
    name: str
    state: str
    ssh: str = ""
    agent: str = ""
    lock: object = field(default_factory=threading.Lock, repr=False)

    def call(self, action, payload=None):
        args = [self.agent, action, "--state", self.state]
        if self.ssh:
            cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
                   "--", self.ssh, shlex.join(["python3"] + args)]
        else:
            cmd = [sys.executable] + args
        body = b"" if payload is None else json.dumps(payload).encode()
        output = subprocess.check_output(cmd, input=body, stderr=subprocess.PIPE, timeout=15)
        return json.loads(output)


def load_nodes(path):
    rules = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rules, dict) or not isinstance(rules.get("nodes"), list):
        raise ValueError("config must contain a nodes list")
    items = rules["nodes"]
    private_state = "~/.local/state/gpu-occupy/dashboard"
    if not 1 <= len(items) <= 32:
        raise ValueError("configure 1 to 32 nodes")
    nodes = []
    for obj in items:
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            raise ValueError("each node needs a name")
        if set(obj) - {"name", "ssh", "agent", "state"}:
            raise ValueError("unknown node field")
        node = Node(obj["name"].strip(), obj.get("state", private_state),
                    obj.get("ssh", ""), obj.get("agent", ""))
        if not node.name or len(node.name) > 80 or any(n.name == node.name for n in nodes):
            raise ValueError("node names must be distinct and 1 to 80 characters")
        if not all(isinstance(s, str) and "\x00" not in s for s in (node.state, node.ssh, node.agent)):
            raise ValueError("state, ssh and agent must be text")
        if not node.state.strip():
            raise ValueError("state path is required")
        if node.ssh and (node.ssh.startswith("-") or any(c.isspace() for c in node.ssh) or not node.agent):
            raise ValueError("SSH nodes require a destination and an agent path")
        if not node.ssh:
            node.agent = str(ASSETS / "agent.py")
        nodes.append(node)
    return nodes


class Backend:
    def __init__(self, nodes):
        self.nodes = {node.name: node for node in nodes}
        self.observed = {node.name: dict(name=node.name, latest=None) for node in nodes}
        self.lock = threading.Lock()
        self.stopped = threading.Event()

    def refresh(self, node):
        # Serialize updates and reads per node to keep revisions in order.
        with node.lock:
            try:
                result = node.call("snapshot")
                with self.lock:
                    self.observed[node.name] = dict(result, name=node.name)
            except (ValueError, OSError, subprocess.SubprocessError):
                with self.lock:
                    self.observed[node.name]["connection_error"] = "采集源连接失败"

    def run(self):
        with ThreadPoolExecutor(max_workers=min(8, len(self.nodes))) as pool:
            while not self.stopped.is_set():
                list(pool.map(self.refresh, self.nodes.values()))
                self.stopped.wait(5)

    def status(self):
        with self.lock:
            return dict(time=int(time.time()), nodes=[dict(n) for n in self.observed.values()])

    def update(self, request):
        if not isinstance(request, dict) or not isinstance(request.get("node"), str):
            return dict(ok=False, status=400, error="invalid node")
        selector = request.pop("node")
        node = self.nodes.get(selector)
        if node is None:
            return dict(ok=False, status=400, error="invalid node")
        with node.lock:
            result = node.call("update", request)
            if result["ok"]:
                with self.lock:
                    self.observed[node.name] = dict(self.observed[node.name], config=result["config"])
            return result


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, port, backend):
        super().__init__(("127.0.0.1", port), Handler)
        self.backend = backend
        self.host = "127.0.0.1:{}".format(self.server_port)
        self.origin = "http://" + self.host


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if args[1] not in ("200", "204", "304"):
            super().log_message(fmt, *args)

    def send_body(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                         "object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, status, data):
        self.send_body(status, json.dumps(data, ensure_ascii=False).encode(), "application/json")

    def has_local_host(self):
        if self.headers.get("Host") != self.server.host:
            self.send_json(403, {"error": "请使用本机看板地址"})
            return False
        return True

    def do_GET(self):
        if not self.has_local_host():
            return
        if self.path == "/api/status":
            self.send_json(200, self.server.backend.status())
        elif self.path in ROUTES:
            filename, content_type = ROUTES[self.path]
            self.send_body(200, (ASSETS / filename).read_bytes(), content_type)
        else:
            self.send_body(404, b"Not found", "text/plain")

    do_HEAD = do_GET

    def do_POST(self):
        if not self.has_local_host():
            return
        if (self.path != "/api/settings" or self.headers.get("Origin") != self.server.origin
                or self.headers.get("Sec-Fetch-Site") not in (None, "same-origin")
                or self.headers.get_content_type() != "application/json"):
            self.send_json(403, {"error": "仅允许从本机看板提交设置"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 18000:
                raise ValueError("invalid body length")
            self.connection.settimeout(5)
            payload = json.loads(self.rfile.read(length))
        except (ValueError, OSError):
            self.send_json(400, {"error": "请求格式无效"})
            return
        try:
            result = self.server.backend.update(payload)
            self.send_json(result.get("status", 200), result)
        except (OSError, subprocess.SubprocessError, ValueError, KeyError):
            self.send_json(503, {"error": "配置更新未确认，请刷新网页查看实际设置"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8801)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        nodes = load_nodes(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    backend = Backend(nodes)
    with DashboardServer(args.port, backend) as server:
        updater = threading.Thread(target=backend.run, daemon=True)
        updater.start()
        print("Dashboard: " + server.origin + "/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            backend.stopped.set()


if __name__ == "__main__":
    main()
