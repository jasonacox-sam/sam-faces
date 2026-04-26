import pytest
import numpy as np
from pathlib import Path
from sam_faces.database import (
    init_db,
    add_person,
    add_encoding,
    get_all_encodings,
    find_person_by_name,
    list_people,
    add_unknown,
    vec_to_blob,
    blob_to_vec,
)


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Use a temporary database for each test."""
    from sam_faces import database

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    init_db()
    yield


def test_add_person():
    pid = add_person("Jane Smith")
    assert len(pid) == 8  # UUID truncated to 8 chars

    # Duplicate name returns same ID
    pid2 = add_person("Jane Smith")
    assert pid == pid2

    # Case insensitive
    pid3 = add_person("jane smith")
    assert pid == pid3


def test_find_person_by_name():
    add_person("John Doe")
    result = find_person_by_name("john doe")
    assert result is not None
    assert result["name"] == "John Doe"

    result = find_person_by_name("Nonexistent")
    assert result is None


def test_add_and_get_encoding():
    pid = add_person("Test Person")
    vector = np.random.rand(128)
    eid = add_encoding(pid, vector, note="test photo", crop_path="/tmp/crop.jpg")

    assert len(eid) == 12  # UUID truncated to 12 chars

    encodings = get_all_encodings()
    assert len(encodings) == 1
    assert encodings[0]["name"] == "Test Person"
    assert encodings[0]["note"] == "test photo"
    assert encodings[0]["crop_path"] == "/tmp/crop.jpg"
    np.testing.assert_array_equal(encodings[0]["vector"], vector)


def test_list_people():
    add_person("Alice")
    add_person("Bob")

    people = list_people()
    assert len(people) == 2
    assert people[0]["encoding_count"] == 0


def test_add_unknown():
    uid = add_unknown("/tmp/photo.jpg", "/tmp/crop.jpg")
    assert len(uid) == 12


def test_vec_to_blob_roundtrip():
    vec = np.array([0.1, 0.2, 0.3, 0.4])
    blob = vec_to_blob(vec)
    vec2 = blob_to_vec(blob)
    np.testing.assert_array_equal(vec, vec2)


def test_database_migration(tmp_path, monkeypatch):
    """Test that pre-v1.0.0 databases without crop_path get migrated."""
    from sam_faces import database

    # Create an old-style database
    old_db = tmp_path / "old.db"
    import sqlite3

    with sqlite3.connect(old_db) as conn:
        conn.executescript("""
            CREATE TABLE people (id TEXT PRIMARY KEY, name TEXT, created_at TEXT);
            CREATE TABLE encodings (id TEXT PRIMARY KEY, person_id TEXT, vector BLOB, note TEXT, added_at TEXT);
        """)

    monkeypatch.setattr(database, "DB_PATH", old_db)
    init_db()

    # Verify crop_path was added
    with sqlite3.connect(old_db) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(encodings)").fetchall()]
        assert "crop_path" in cols
