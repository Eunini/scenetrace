"""SQLite metadata; all user-owned queries require an owner identifier."""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, filename TEXT NOT NULL,
 state TEXT NOT NULL, duration REAL DEFAULT 0, width INTEGER DEFAULT 0,
 height INTEGER DEFAULT 0, bytes INTEGER NOT NULL, created REAL NOT NULL, error TEXT
);
CREATE INDEX IF NOT EXISTS videos_owner ON videos(owner);
CREATE TABLE IF NOT EXISTS frames (
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, video_id TEXT NOT NULL REFERENCES videos(id)
 ON DELETE CASCADE, seconds REAL NOT NULL, sharpness REAL NOT NULL, embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, video_id TEXT, kind TEXT NOT NULL,
 state TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, result TEXT, error TEXT,
 created REAL NOT NULL, updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, video_id TEXT NOT NULL REFERENCES videos(id)
 ON DELETE CASCADE, start REAL NOT NULL, end REAL NOT NULL, state TEXT NOT NULL,
 job_id TEXT, created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, query TEXT NOT NULL, plan TEXT NOT NULL,
 response TEXT NOT NULL, created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, turn_id TEXT NOT NULL REFERENCES turns(id)
 ON DELETE CASCADE, vote INTEGER NOT NULL, comment TEXT NOT NULL, created REAL NOT NULL
);
"""


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = root / "workspace.sqlite3"
        with self.connect() as db:
            db.executescript(SCHEMA)
            # Interrupted operations cannot silently look complete after a restart.
            db.execute("UPDATE jobs SET state='failed', error='Host restarted; please retry.' "
                       "WHERE state IN ('queued','running')")
            db.execute("UPDATE videos SET state='failed', error='Indexing interrupted; upload again.' "
                       "WHERE state='indexing'")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def one(self, sql, args=()):
        with self.connect() as db:
            row = db.execute(sql, args).fetchone()
        return dict(row) if row else None

    def all(self, sql, args=()):
        with self.connect() as db:
            return [dict(r) for r in db.execute(sql, args).fetchall()]

    def execute(self, sql, args=()):
        with self.connect() as db:
            return db.execute(sql, args).rowcount

    def library(self, owner, mode="list", contains=""):
        if mode == "count":
            return self.one(
                "SELECT COUNT(*) AS videos, COALESCE(SUM(duration),0) AS duration, "
                "COALESCE(SUM(bytes),0) AS bytes FROM videos WHERE owner=? AND state='ready'",
                (owner,),
            )
        # No model-generated SQL is executed. LIKE metacharacters have literal meaning.
        escaped = contains.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return self.all("SELECT * FROM videos WHERE owner=? AND filename LIKE ? ESCAPE '\\' "
                        "ORDER BY created DESC LIMIT 20", (owner, f"%{escaped}%"))

    def update_job(self, job_id, state, progress=0, result=None, error=None):
        self.execute("UPDATE jobs SET state=?,progress=?,result=?,error=?,updated=? WHERE id=?",
                     (state, progress, json.dumps(result) if result is not None else None,
                      error, time.time(), job_id))
