"""
Tests for sam_faces/enroll.py — face enrollment into the people database.

enroll() detects a face with face_recognition, generates a 128-dim encoding,
writes a person/encoding row into SQLite, and saves an audit-trail crop
thumbnail. These tests mock face_recognition (so no dlib models are needed),
redirect the SQLite DB and crops directory into tmp_path, and verify the
database writes and crop-file side effects.

Test categories covered (per project convention):
  Security, Performance, Retry, Unit, Integration, Functional, Frame
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

import sam_faces.enroll as enroll_mod
from sam_faces.enroll import enroll
from sam_faces.database import get_all_encodings, list_people

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def temp_env(tmp_path, monkeypatch):
    """Redirect the SQLite DB and crops directory into a temp location so each
    test is fully isolated with no real filesystem/DB side effects."""
    from sam_faces import database

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "people.db")
    monkeypatch.setattr(enroll_mod, "CROPS_DIR", tmp_path / "crops")
    yield


def _fake_image(w=200, h=200):
    """A valid HxWx3 uint8 array, as face_recognition.load_image_file returns."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _patch_faces(locations, encodings, image=None):
    """Context managers patching the three face_recognition calls enroll uses."""
    if image is None:
        image = _fake_image()
    return (
        patch(
            "sam_faces.enroll.face_recognition.load_image_file",
            return_value=image,
        ),
        patch(
            "sam_faces.enroll.face_recognition.face_locations",
            return_value=locations,
        ),
        patch(
            "sam_faces.enroll.face_recognition.face_encodings",
            return_value=encodings,
        ),
    )


def _photo(tmp_path, name="face.jpg"):
    p = tmp_path / name
    p.write_bytes(b"not-really-a-jpeg")  # content is irrelevant; load is mocked
    return p


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


def test_security_missing_file_raises_before_any_write(tmp_path):
    """A nonexistent photo path is rejected with FileNotFoundError before the
    DB is touched — no partial person/encoding rows are created."""
    with pytest.raises(FileNotFoundError):
        enroll("Mallory", str(tmp_path / "does_not_exist.jpg"))

    # DB must remain empty.
    assert list_people() == []
    assert get_all_encodings() == []


def test_security_crop_stays_inside_crops_dir(tmp_path):
    """The audit-trail thumbnail is written only under the configured CROPS_DIR
    (never escaping via the encoding id / note)."""
    photo = _photo(tmp_path)
    p1, p2, p3 = _patch_faces([(10, 100, 100, 10)], [np.random.rand(128)])
    with p1, p2, p3:
        result = enroll("Jane", str(photo), note="../../etc/passwd")

    crop = Path(result["crop_path"])
    assert crop.exists()
    assert crop.parent == (tmp_path / "crops")


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_performance_reenroll_reuses_person_no_duplicate_rows(tmp_path):
    """Enrolling the same name twice reuses the existing person row (one
    person, two encodings) rather than exploding the people table."""
    photo = _photo(tmp_path)
    for _ in range(2):
        p1, p2, p3 = _patch_faces([(10, 100, 100, 10)], [np.random.rand(128)])
        with p1, p2, p3:
            enroll("Repeat Person", str(photo))

    people = list_people()
    assert len(people) == 1
    assert people[0]["encoding_count"] == 2


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="N/A: enroll() is a single deterministic pass over one photo with "
    "no network/IO retry semantics to exercise."
)
def test_retry_not_applicable():
    """Placeholder — enroll has no retry logic."""
    pass


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------


def test_unit_single_face_return_contract(tmp_path):
    """A single-face enroll returns the documented dict and writes the crop."""
    photo = _photo(tmp_path, name="jane.jpg")
    p1, p2, p3 = _patch_faces([(10, 100, 100, 10)], [np.random.rand(128)])
    with p1, p2, p3:
        result = enroll("Jane Smith", str(photo))

    assert set(result) == {
        "encoding_id",
        "person_id",
        "person_name",
        "crop_path",
        "note",
    }
    assert result["person_name"] == "Jane Smith"
    # note defaults to the source filename when not supplied
    assert result["note"] == "jane.jpg"
    # crop was renamed to <encoding_id>.jpg under the crops dir
    assert result["crop_path"].endswith(f"{result['encoding_id']}.jpg")
    assert Path(result["crop_path"]).exists()


def test_unit_custom_note_is_stored(tmp_path):
    """An explicit note overrides the filename default."""
    photo = _photo(tmp_path)
    p1, p2, p3 = _patch_faces([(10, 100, 100, 10)], [np.random.rand(128)])
    with p1, p2, p3:
        result = enroll("Jane", str(photo), note="found at the office party")

    assert result["note"] == "found at the office party"


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def test_integration_encoding_persisted_and_retrievable(tmp_path):
    """The enrolled encoding round-trips through SQLite: it is retrievable via
    get_all_encodings with the correct name, vector, and final crop_path."""
    photo = _photo(tmp_path)
    vector = np.random.rand(128)
    p1, p2, p3 = _patch_faces([(10, 100, 100, 10)], [vector])
    with p1, p2, p3:
        result = enroll("Persisted Person", str(photo))

    rows = get_all_encodings()
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "Persisted Person"
    np.testing.assert_allclose(row["vector"], vector, rtol=1e-6)
    # DB crop_path was updated to the final <eid>.jpg (not the temp file)
    assert row["crop_path"] == result["crop_path"]
    assert Path(row["crop_path"]).exists()


# ---------------------------------------------------------------------------
# Functional
# ---------------------------------------------------------------------------


def test_functional_multiface_requires_index(tmp_path):
    """With multiple faces and no face_index, enroll refuses and explains."""
    photo = _photo(tmp_path)
    locs = [(10, 100, 100, 10), (10, 200, 100, 110)]
    encs = [np.random.rand(128), np.random.rand(128)]
    p1, p2, p3 = _patch_faces(locs, encs)
    with p1, p2, p3:
        with pytest.raises(ValueError, match="Multiple faces"):
            enroll("Ambiguous", str(photo))


def test_functional_multiface_index_selects_correct_encoding(tmp_path):
    """When face_index is given, the selected face's encoding is the one
    stored."""
    photo = _photo(tmp_path)
    locs = [(10, 100, 100, 10), (10, 200, 100, 110)]
    enc0 = np.full(128, 0.1)
    enc1 = np.full(128, 0.9)
    p1, p2, p3 = _patch_faces(locs, [enc0, enc1])
    with p1, p2, p3:
        enroll("Second Person", str(photo), face_index=1)

    rows = get_all_encodings()
    assert len(rows) == 1
    np.testing.assert_allclose(rows[0]["vector"], enc1, rtol=1e-6)


# ---------------------------------------------------------------------------
# Frame (boundary / edge conditions)
# ---------------------------------------------------------------------------


def test_frame_no_faces_detected_raises(tmp_path):
    """Boundary: an empty encodings list means no face was found."""
    photo = _photo(tmp_path)
    p1, p2, p3 = _patch_faces([], [])
    with p1, p2, p3:
        with pytest.raises(ValueError, match="No faces detected"):
            enroll("Nobody", str(photo))

    assert get_all_encodings() == []


def test_frame_corner_face_crop_padding_clamped(tmp_path):
    """Boundary: a face flush against the top-left corner would push the padded
    crop box to negative coordinates; enroll must clamp and still succeed."""
    photo = _photo(tmp_path)
    # loc = (top, right, bottom, left) hugging the origin
    p1, p2, p3 = _patch_faces(
        [(0, 30, 30, 0)], [np.random.rand(128)], image=_fake_image(40, 40)
    )
    with p1, p2, p3:
        result = enroll("Corner", str(photo))

    assert Path(result["crop_path"]).exists()
    assert len(get_all_encodings()) == 1
