import sqlite3
import numpy as np
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Default database location — uses existing Sam DB if present
DEFAULT_DB_DIR = Path.home() / ".openclaw" / "workspace" / "faces"
DB_PATH = DEFAULT_DB_DIR / "people.db"


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
                added_at   TEXT NOT NULL,
                crop_path  TEXT
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


def vec_to_blob(encoding: np.ndarray) -> bytes:
    return encoding.astype(np.float64).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float64)


def add_person(name: str) -> str:
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


def add_encoding(person_id: str, encoding: np.ndarray, note: str = "", crop_path: str = "") -> str:
    eid = str(uuid.uuid4())[:12]
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO encodings (id, person_id, vector, note, added_at, crop_path) VALUES (?, ?, ?, ?, ?, ?)",
            (eid, person_id, vec_to_blob(encoding), note, datetime.now(timezone.utc).isoformat(), crop_path)
        )
    return eid


def get_all_encodings() -> list[dict]:
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
