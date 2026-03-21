#!/usr/bin/env python3
"""
face_db.py — SQLite database layer for Sam's face recognition system

Schema:
  people(id TEXT PK, name TEXT, created_at TEXT)
  encodings(id TEXT PK, person_id TEXT FK, vector BLOB, note TEXT, added_at TEXT)
  unknown_candidates(id TEXT PK, image_path TEXT, face_crop_path TEXT,
                     detected_at TEXT, resolved INTEGER DEFAULT 0,
                     resolved_as TEXT)
"""

import os
import sqlite3
import numpy as np
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.environ.get("SAM_FACES_DB", Path(__file__).parent.parent / "faces" / "people.db"))

# ── Connection ─────────────────────────────────────────────────────────────

def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS people (
                id         TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS encodings (
                id         TEXT PRIMARY KEY,
                person_id  TEXT NOT NULL REFERENCES people(id),
                vector     BLOB NOT NULL,
                note       TEXT,
                added_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS unknown_candidates (
                id             TEXT PRIMARY KEY,
                image_path     TEXT NOT NULL,
                face_crop_path TEXT,
                detected_at    TEXT NOT NULL,
                resolved       INTEGER DEFAULT 0,
                resolved_as    TEXT
            );
        """)
    print(f"✅ Database ready: {DB_PATH}")

# ── Encoding helpers ────────────────────────────────────────────────────────

def vec_to_blob(encoding: np.ndarray) -> bytes:
    return encoding.astype(np.float64).tobytes()

def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float64)

# ── People ──────────────────────────────────────────────────────────────────

def add_person(name: str) -> str:
    """Add a new person, return their id. Returns existing id if name matches."""
    existing = find_person_by_name(name)
    if existing:
        return existing["id"]
    pid = str(uuid.uuid4())[:8]
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO people (id, name, created_at) VALUES (?, ?, ?)",
            (pid, name, datetime.now(timezone.utc).isoformat())
        )
    return pid

def find_person_by_name(name: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM people WHERE LOWER(name) = LOWER(?)", (name,)
        ).fetchone()
    return dict(row) if row else None

def list_people() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT p.id, p.name, p.created_at, COUNT(e.id) as encoding_count
            FROM people p
            LEFT JOIN encodings e ON e.person_id = p.id
            GROUP BY p.id
            ORDER BY p.name
        """).fetchall()
    return [dict(r) for r in rows]

# ── Encodings ───────────────────────────────────────────────────────────────

def add_encoding(person_id: str, encoding: np.ndarray, note: str = "") -> str:
    eid = str(uuid.uuid4())[:12]
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO encodings (id, person_id, vector, note, added_at) VALUES (?, ?, ?, ?, ?)",
            (eid, person_id, vec_to_blob(encoding), note, datetime.now(timezone.utc).isoformat())
        )
    return eid

def get_all_encodings() -> list[dict]:
    """Return all encodings with person name attached."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT e.id, e.person_id, e.vector, e.note, e.added_at, p.name
            FROM encodings e
            JOIN people p ON p.id = e.person_id
        """).fetchall()
    return [
        {**dict(r), "vector": blob_to_vec(r["vector"])}
        for r in rows
    ]

# ── Unknown candidates ───────────────────────────────────────────────────────

def add_unknown(image_path: str, face_crop_path: str = "") -> str:
    uid = str(uuid.uuid4())[:12]
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO unknown_candidates (id, image_path, face_crop_path, detected_at) VALUES (?, ?, ?, ?)",
            (uid, image_path, face_crop_path, datetime.now(timezone.utc).isoformat())
        )
    return uid

def resolve_unknown(unknown_id: str, person_name: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE unknown_candidates SET resolved=1, resolved_as=? WHERE id=?",
            (person_name, unknown_id)
        )

def list_unknowns(unresolved_only: bool = True) -> list[dict]:
    with get_conn() as conn:
        query = "SELECT * FROM unknown_candidates"
        if unresolved_only:
            query += " WHERE resolved=0"
        query += " ORDER BY detected_at DESC"
        rows = conn.execute(query).fetchall()
    return [dict(r) for r in rows]

# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    init_db()
    if "--list" in sys.argv:
        people = list_people()
        if not people:
            print("No people enrolled yet.")
        for p in people:
            print(f"  {p['name']:30s} — {p['encoding_count']} encoding(s)  [{p['id']}]")
    elif "--unknowns" in sys.argv:
        unknowns = list_unknowns()
        if not unknowns:
            print("No unresolved unknown faces.")
        for u in unknowns:
            print(f"  [{u['id']}] {u['detected_at'][:10]}  {u['image_path']}")
