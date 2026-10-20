"""SQLite persistence with transactional event deduplication and optimistic concurrency."""
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Conflict(Exception):
    pass


class DispatchStore:
    def __init__(self, path=None):
        self.path = Path(path or os.getenv("DISPATCH_DB_PATH", Path(__file__).resolve().parents[1] / "runtime/dispatch.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS dispatches (id TEXT PRIMARY KEY, version INTEGER NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS events (dispatch_id TEXT NOT NULL, event_id TEXT NOT NULL, request TEXT NOT NULL, PRIMARY KEY(dispatch_id,event_id))")

    @contextmanager
    def connection(self):
        with sqlite3.connect(self.path, timeout=10) as db:
            db.row_factory = sqlite3.Row
            yield db

    def create(self, data):
        with self.connection() as db:
            db.execute("INSERT INTO dispatches VALUES (?,?,?,?)", (data["dispatch_id"], data["version"], json.dumps(data, allow_nan=False), data["created_at"]))

    def get(self, dispatch_id):
        with self.connection() as db:
            row = db.execute("SELECT payload FROM dispatches WHERE id=?", (dispatch_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def recent(self):
        with self.connection() as db:
            rows = db.execute("SELECT payload FROM dispatches ORDER BY created_at DESC LIMIT 20").fetchall()
            return [json.loads(row[0]) for row in rows]

    def apply_event(self, dispatch_id, event, mutate):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM dispatches WHERE id=?", (dispatch_id,)).fetchone()
            if row is None:
                raise KeyError("Dispatch not found")
            data = json.loads(row[0])
            existing = db.execute("SELECT request FROM events WHERE dispatch_id=? AND event_id=?", (dispatch_id, event.event_id)).fetchone()
            event_json = event.model_dump_json()
            if existing:
                if existing[0] != event_json:
                    raise Conflict("Event ID already used for a different action")
                return data
            if data["version"] != event.expected_version:
                raise Conflict("Dispatch changed on another device. Refresh and try again.")
            mutate(data, event)
            data["version"] += 1
            db.execute("UPDATE dispatches SET version=?,payload=? WHERE id=?", (data["version"], json.dumps(data, allow_nan=False), dispatch_id))
            db.execute("INSERT INTO events VALUES (?,?,?)", (dispatch_id, event.event_id, event_json))
            return data
