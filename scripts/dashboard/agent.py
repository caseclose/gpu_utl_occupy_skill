"""Run a local NVIDIA telemetry collector or operate on its private state."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from collector import publish, supervise


def read(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def rows(path, max_bytes):
    try:
        with path.open("rb") as stream:
            size = path.stat().st_size
            stream.seek(max(0, size - max_bytes))
            if size > max_bytes:
                stream.readline()
            for line in stream:
                try:
                    yield json.loads(line.decode("utf-8"))
                except (UnicodeError, ValueError):
                    continue
    except FileNotFoundError:
        return


def snapshot(directory):
    now = int(time.time())
    metrics, events = [], []
    for path in sorted(directory.glob("metrics-*.ndjson"))[-2:]:
        metrics.extend(row for row in rows(path, 2000000) if row["t"] >= now - 3600)
    for path in sorted(directory.glob("events-*.ndjson"))[-8:]:
        events.extend(rows(path, 128000))
    return dict(latest=read(directory / "latest.json"), metrics=metrics, events=events[-250:],
                config=read(directory / "config.json"), health=read(directory / "health.json"))


def validate(payload):
    if not isinstance(payload, dict) or set(payload) != {"cards", "enabled", "target", "revision"}:
        raise ValueError("invalid fields")
    cards, enabled, target, rev = (payload[k] for k in ("cards", "enabled", "target", "revision"))
    if (not isinstance(cards, list) or not 1 <= len(cards) <= 128
            or any(not isinstance(card, str) for card in cards) or len(set(cards)) != len(cards)):
        raise ValueError("invalid GPU UUID list")
    if type(enabled) is not bool or type(target) is not int or not 1 <= target <= 100:
        raise ValueError("目标利用率必须为 1–100 的整数")
    if type(rev) is not int or rev < 1:
        raise ValueError("invalid revision")
    return cards, enabled, target, rev


def update(directory, payload):
    cards, enabled, target, rev = validate(payload)
    path = directory / "config.json"
    with (directory / "config.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        latest = read(directory / "latest.json")
        if not latest or not latest["controllable"] or time.time() - latest["time"] > 30:
            return dict(ok=False, status=409, error="节点未启用控制或采集数据已过期")
        if set(cards) - {device["uuid"] for device in latest["gpus"]}:
            raise ValueError("选中的 GPU 已不在节点中")
        config = read(path)
        if config is None or config["revision"] != rev:
            return dict(ok=False, status=409, error="设置已更新，请刷新后重试")
        for card in cards:
            config["cards"][card] = dict(enabled=enabled, target=target)
        now = int(time.time())
        config.update(revision=rev + 1, time=now)
        publish(path, config)
        with (directory / "settings.ndjson").open("a", encoding="utf-8") as log:
            log.write(json.dumps(dict(payload, time=now, revision=rev + 1)) + "\n")
        return dict(ok=True, config=config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "snapshot", "update"))
    parser.add_argument("--state", type=Path, default=Path.home() / ".local/state/gpu-occupy/dashboard")
    parser.add_argument("--pulse", type=Path, help="optional compiled pulse executable for control")
    args = parser.parse_args()
    directory = args.state.expanduser().resolve()
    os.umask(0o077)
    if args.action == "collect":
        pulse = args.pulse.expanduser().resolve() if args.pulse else None
        if pulse and (not pulse.is_file() or not os.access(str(pulse), os.X_OK)):
            parser.error("--pulse must point to an executable file")
        supervise(directory, pulse)
        return
    if args.action == "snapshot":
        result = snapshot(directory)
    else:
        try:
            result = update(directory, json.load(sys.stdin))
        except (ValueError, TypeError, OSError) as exc:
            result = dict(ok=False, status=400, error=str(exc))
    # SSH transports can run with a non-UTF-8 locale.
    print(json.dumps(result))


if __name__ == "__main__":
    main()
