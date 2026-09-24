from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    archive_root TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    phash TEXT NOT NULL,
    crop_hash TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    first_seen_ts REAL NOT NULL,
    indexed_ts REAL NOT NULL,
    embedding BLOB,
    embedding_dim INTEGER,
    embedding_model TEXT
);

CREATE INDEX IF NOT EXISTS idx_images_sha256 ON images(sha256);
CREATE INDEX IF NOT EXISTS idx_images_archive_root ON images(archive_root);
CREATE INDEX IF NOT EXISTS idx_images_first_seen ON images(first_seen_ts);
"""


class PhotoIndex:
    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def get_by_path(self, path: Path | str):
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM images WHERE path = ?",
                (str(Path(path).resolve()),),
            ).fetchone()

    def exact(self, sha256: str):
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM images WHERE sha256 = ? ORDER BY first_seen_ts",
                (sha256,),
            ).fetchall()

    def all_rows(self):
        with self.connect() as conn:
            return conn.execute("SELECT * FROM images ORDER BY id").fetchall()

    def upsert(self, record: dict) -> None:
        columns = [
            "path",
            "archive_root",
            "sha256",
            "phash",
            "crop_hash",
            "width",
            "height",
            "size_bytes",
            "mtime_ns",
            "first_seen_ts",
            "indexed_ts",
            "embedding",
            "embedding_dim",
            "embedding_model",
        ]
        values = [record.get(c) for c in columns]
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(
            f"{c}=excluded.{c}"
            for c in columns
            if c not in {"path", "first_seen_ts"}
        )
        sql = f"""
        INSERT INTO images ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT(path) DO UPDATE SET
            {updates},
            first_seen_ts = MIN(images.first_seen_ts, excluded.first_seen_ts)
        """
        with self.connect() as conn:
            conn.execute(sql, values)
            conn.commit()

    def prune_root(self, root: Path | str, existing_paths: Iterable[str]) -> int:
        root_s = str(Path(root).resolve())
        keep = set(existing_paths)
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, path FROM images WHERE archive_root = ?",
                (root_s,),
            ).fetchall()
            stale = [row["id"] for row in rows if row["path"] not in keep]
            if stale:
                conn.executemany(
                    "DELETE FROM images WHERE id = ?",
                    [(image_id,) for image_id in stale],
                )
                conn.commit()
            return len(stale)
