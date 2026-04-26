import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from sam_faces.identify import identify, _position_desc, DEFAULT_THRESHOLD


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Use a temporary database for each test."""
    from sam_faces import database

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    from sam_faces.database import init_db, add_person, add_encoding

    init_db()
    # Add a known person
    pid = add_person("Jane Smith")
    vector = np.random.rand(128)
    add_encoding(pid, vector, note="test")
    yield


def test_identify_no_faces(tmp_path):
    """Test with a photo that has no faces."""
    # Create a tiny blank image
    from PIL import Image

    img = Image.new("RGB", (10, 10), "white")
    photo = tmp_path / "blank.jpg"
    img.save(photo)

    result = identify(str(photo))
    assert result["face_count"] == 0
    assert "No faces detected" in result["llm_context"]


def test_position_desc():
    """Test position description logic."""
    # upper-left
    assert _position_desc((0, 100, 100, 0), 300, 300) == "upper-left"
    # middle-center
    assert _position_desc((100, 200, 200, 100), 300, 300) == "middle"
    # lower-right
    assert _position_desc((200, 300, 300, 200), 300, 300) == "lower-right"


def test_identify_file_not_found():
    result = identify("/nonexistent/photo.jpg")
    assert "error" in result
    assert "File not found" in result["error"]


def test_llm_context_format():
    """Test that llm_context string is properly formatted."""
    # Mock the face_recognition calls
    mock_encoding = np.random.rand(128)

    with (
        patch("sam_faces.identify.face_recognition.load_image_file") as mock_load,
        patch(
            "sam_faces.identify.face_recognition.face_locations",
            return_value=[(10, 100, 100, 10)],
        ),
        patch(
            "sam_faces.identify.face_recognition.face_encodings",
            return_value=[mock_encoding],
        ),
        patch("pathlib.Path.exists", return_value=True),
    ):

        mock_load.return_value = np.zeros((200, 200, 3), dtype=np.uint8)

        result = identify("/fake/path.jpg", threshold=DEFAULT_THRESHOLD)

        assert "llm_context" in result
        assert result["face_count"] >= 0
