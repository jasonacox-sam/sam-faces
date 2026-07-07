"""
sam_faces/database.py — face database operations.

Backend is selected by the SAM_FACES_DB environment variable:
  • unset or a filesystem path        → SQLite  (default: ~/.openclaw/workspace/faces/people.db)
  • a postgres:// / postgresql:// URL → PostgreSQL (requires the `postgres` extra: psycopg2)

Both backends use an identical schema and a byte-compatible vector encoding
(float64.tobytes()), so a database is portable between them.

Tables:
  people(id TEXT PK, name TEXT, created_at TEXT)
  encodings(id TEXT PK, person_id TEXT FK, vector BLOB/BYTEA, note TEXT, added_at TEXT, crop_path TEXT)
  unknown_candidates(id TEXT PK, image_path TEXT, face_crop_path TEXT, detected_at TEXT,
                     resolved INTEGER DEFAULT 0, resolved_as TEXT)
"""

import os
import uuid
import numpy as np
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# Default database location — uses existing Sam DB if present
DEFAULT_DB_DIR = Path.home() / ".openclaw" / "workspace" / "faces"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "people.db"

_DB = os.environ.get("SAM_FACES_DB", str(DEFAULT_DB_PATH))
IS_POSTGRES = _DB.startswith(("postgres://", "postgresql://"))

# Query placeholder differs by driver: %s (psycopg2) vs ? (sqlite3)
_PH = "%s" if IS_POSTGRES else "?"
# Backward-compatible export for SQLite callers (None when using Postgres)
DB_PATH = None if IS_POSTGRES else Path(_DB)

if IS_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    def _bin(b: bytes):
        return psycopg2.Binary(b)
else:
    import sqlite3

    def _bin(b: bytes):
        return b


def get_conn():
    """Open a new connection to the configured backend."""
    if IS_POSTGRES:
        return psycopg2.connect(_DB, cursor_factory=RealDictCursor)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _cursor(commit: bool = False):
    """Yield a cursor over a fresh connection; commit (optional) and close on exit.

    Rows are dict-convertible on both backends (sqlite3.Row / psycopg2 RealDictRow)."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        yield cur
        if commit:
            conn.commit()
    finally:
        try:
            cur.close()
        finally:
            conn.close()


def init_db():
    """Create tables if they don't exist, and migrate schema if needed."""
    blob = "BYTEA" if IS_POSTGRES else "BLOB"
    tables = [
        "CREATE TABLE IF NOT EXISTS people ("
        "id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS encodings ("
        "id TEXT PRIMARY KEY, person_id TEXT NOT NULL REFERENCES people(id), "
        f"vector {blob} NOT NULL, note TEXT, added_at TEXT NOT NULL, crop_path TEXT)",
        "CREATE TABLE IF NOT EXISTS unknown_candidates ("
        "id TEXT PRIMARY KEY, image_path TEXT NOT NULL, face_crop_path TEXT, "
        "detected_at TEXT NOT NULL, resolved INTEGER DEFAULT 0, resolved_as TEXT)",
    ]
    with _cursor(commit=True) as cur:
        for ddl in tables:
            cur.execute(ddl)
        # Migrate: add crop_path to encodings if missing (pre-v1.0.0 databases)
        if IS_POSTGRES:
            cur.execute("ALTER TABLE encodings ADD COLUMN IF NOT EXISTS crop_path TEXT")
        else:
            cols = [r[1] for r in cur.execute("PRAGMA table_info(encodings)").fetchall()]
            if "crop_path" not in cols:
                cur.execute("ALTER TABLE encodings ADD COLUMN crop_path TEXT")


def vec_to_blob(encoding: np.ndarray) -> bytes:
    return encoding.astype(np.float64).tobytes()


def blob_to_vec(blob) -> np.ndarray:
    return np.frombuffer(bytes(blob), dtype=np.float64)


def add_person(name: str) -> str:
    existing = find_person_by_name(name)
    if existing:
        return existing["id"]
    pid = str(uuid.uuid4())[:8]
    with _cursor(commit=True) as cur:
        cur.execute(
            f"INSERT INTO people (id, name, created_at) VALUES ({_PH}, {_PH}, {_PH})",
            (pid, name, datetime.now(timezone.utc).isoformat()),
        )
    return pid


def find_person_by_name(name: str) -> dict | None:
    with _cursor() as cur:
        cur.execute(f"SELECT * FROM people WHERE LOWER(name) = LOWER({_PH})", (name,))
        row = cur.fetchone()
    return dict(row) if row else None


def list_people() -> list[dict]:
    with _cursor() as cur:
        cur.execute(
            "SELECT p.id, p.name, p.created_at, COUNT(e.id) AS encoding_count "
            "FROM people p LEFT JOIN encodings e ON e.person_id = p.id "
            "GROUP BY p.id, p.name, p.created_at ORDER BY p.name"
        )
        return [dict(r) for r in cur.fetchall()]


def add_encoding(person_id: str, encoding: np.ndarray, note: str = "", crop_path: str = "") -> str:
    eid = str(uuid.uuid4())[:12]
    with _cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO encodings (id, person_id, vector, note, added_at, crop_path) "
            f"VALUES ({_PH}, {_PH}, {_PH}, {_PH}, {_PH}, {_PH})",
            (eid, person_id, _bin(vec_to_blob(encoding)), note,
             datetime.now(timezone.utc).isoformat(), crop_path),
        )
    return eid


def get_all_encodings() -> list[dict]:
    with _cursor() as cur:
        cur.execute(
            "SELECT e.id, e.person_id, e.vector, e.note, e.added_at, e.crop_path, p.name "
            "FROM encodings e JOIN people p ON p.id = e.person_id"
        )
        return [{**dict(r), "vector": blob_to_vec(r["vector"])} for r in cur.fetchall()]


def update_crop_path(encoding_id: str, crop_path: str):
    with _cursor(commit=True) as cur:
        cur.execute(f"UPDATE encodings SET crop_path={_PH} WHERE id={_PH}", (crop_path, encoding_id))


def add_unknown(image_path: str, face_crop_path: str = "") -> str:
    uid = str(uuid.uuid4())[:12]
    with _cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO unknown_candidates (id, image_path, face_crop_path, detected_at) "
            f"VALUES ({_PH}, {_PH}, {_PH}, {_PH})",
            (uid, image_path, face_crop_path, datetime.now(timezone.utc).isoformat()),
        )
    return uid


def resolve_unknown(unknown_id: str, person_name: str):
    with _cursor(commit=True) as cur:
        cur.execute(
            f"UPDATE unknown_candidates SET resolved=1, resolved_as={_PH} WHERE id={_PH}",
            (person_name, unknown_id),
        )


def list_unknowns(unresolved_only: bool = True) -> list[dict]:
    query = "SELECT * FROM unknown_candidates"
    if unresolved_only:
        query += " WHERE resolved=0"
    query += " ORDER BY detected_at DESC"
    with _cursor() as cur:
        cur.execute(query)
        return [dict(r) for r in cur.fetchall()]
