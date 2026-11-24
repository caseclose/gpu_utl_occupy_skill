"""Persist one dashboard's display preferences independently of device settings."""

import fcntl
import json
from pathlib import Path
import tempfile


class ViewPreferences:
    def __init__(self, path, nodes):
        self.path = path
        self.nodes = tuple(nodes)
        self.lock_path = path.with_name(path.name + ".lock")

    def read(self):
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return dict(revision=0, order=[], notes={})
        if (
            not isinstance(value, dict)
            or type(value.get("revision")) is not int
            or value["revision"] < 0
            or not isinstance(value.get("order"), list)
            or any(not isinstance(node, str) for node in value["order"])
            or not isinstance(value.get("notes"), dict)
            or any(not isinstance(note, str) for note in value["notes"].values())
        ):
            raise ValueError("显示偏好文件格式无效")
        return value

    @staticmethod
    def clean_note(note):
        if not isinstance(note, str) or len(note) > 40:
            raise ValueError("备注最多 40 个字符")
        note.encode("utf-8")
        return note.strip()

    def update(self, payload):
        fields = set(payload)
        with self.lock_path.open("a", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            preferences = self.read()
            if fields == {"import"}:
                if preferences["revision"]:
                    return preferences
                order, notes = self.import_legacy(payload["import"])
                if not order and not notes:
                    return preferences
                preferences.update(order=order, notes=notes)
            else:
                node = payload.get("node")
                if node not in self.nodes:
                    raise ValueError("节点无效")
                if fields == {"node", "note"}:
                    note = self.clean_note(payload["note"])
                    if note:
                        preferences["notes"][node] = note
                    else:
                        preferences["notes"].pop(node, None)
                elif fields == {"node", "destination"}:
                    destination = payload["destination"]
                    if destination not in self.nodes or destination == node:
                        raise ValueError("节点排序无效")
                    order = list(dict.fromkeys([*preferences["order"], *self.nodes]))
                    index = order.index(destination)
                    order.remove(node)
                    order.insert(index, node)
                    preferences["order"] = order
                else:
                    raise ValueError("显示偏好格式无效")
            preferences["revision"] += 1
            self.write(preferences)
            return preferences

    def import_legacy(self, value):
        if not isinstance(value, dict) or set(value) != {"order", "notes"}:
            raise ValueError("旧版显示偏好格式无效")
        if not isinstance(value["order"], list) or not isinstance(value["notes"], dict):
            raise ValueError("旧版显示偏好格式无效")
        order = list(
            dict.fromkeys(node for node in value["order"] if node in self.nodes)
        )
        notes = {}
        for node, raw in value["notes"].items():
            if node in self.nodes:
                note = self.clean_note(raw)
                if note:
                    notes[node] = note
        return order, notes

    def write(self, value):
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.path.parent,
            prefix=".view-preferences.",
            encoding="utf-8",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(value, stream, ensure_ascii=False)
                stream.close()
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)
