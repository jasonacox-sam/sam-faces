import pytest
from unittest.mock import patch, MagicMock
from sam_faces.cli import main
import sys


def test_cli_help(capsys):
    """Test that CLI prints help when no args given."""
    with pytest.raises(SystemExit) as exc:
        with patch.object(sys, "argv", ["sam-faces"]):
            main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "usage" in captured.out.lower()


def test_cli_identify():
    """Test identify command."""
    mock_result = {
        "face_count": 1,
        "faces": [{"name": "Jane", "confidence": 0.9}],
        "llm_context": "1 face detected: Jane",
    }

    with (
        patch("sam_faces.cli.identify", return_value=mock_result) as mock_identify,
        patch.object(sys, "argv", ["sam-faces", "identify", "photo.jpg"]),
    ):
        main()
        mock_identify.assert_called_once_with(
            "photo.jpg", threshold=0.55, save_unknowns=True, save_crops=True
        )


def test_cli_enroll():
    """Test enroll command."""
    mock_result = {
        "encoding_id": "abc123",
        "person_id": "person456",
        "person_name": "Jane Smith",
        "crop_path": "/tmp/crop.jpg",
        "note": "test",
    }

    with (
        patch("sam_faces.cli.enroll", return_value=mock_result) as mock_enroll,
        patch.object(
            sys,
            "argv",
            ["sam-faces", "enroll", "--name", "Jane Smith", "--photo", "photo.jpg"],
        ),
    ):
        main()
        mock_enroll.assert_called_once_with("Jane Smith", "photo.jpg", "", None)


def test_cli_enroll_with_note_and_index():
    """Test enroll command with optional args."""
    with (
        patch("sam_faces.cli.enroll", return_value={}) as mock_enroll,
        patch.object(
            sys,
            "argv",
            [
                "sam-faces",
                "enroll",
                "--name",
                "Jane",
                "--photo",
                "p.jpg",
                "--note",
                "birthday",
                "--face-index",
                "1",
            ],
        ),
    ):
        main()
        mock_enroll.assert_called_once_with("Jane", "p.jpg", "birthday", 1)


def test_cli_legacy_photo_flag():
    """Test backward compatibility with --photo flag."""
    mock_result = {"face_count": 2, "faces": [], "llm_context": "test"}

    with (
        patch("sam_faces.cli.identify", return_value=mock_result) as mock_identify,
        patch.object(sys, "argv", ["sam-faces", "--photo", "photo.jpg"]),
    ):
        main()
        mock_identify.assert_called_once_with(
            "photo.jpg", threshold=0.55, save_unknowns=True, save_crops=True
        )


def test_cli_list(capsys):
    """Test list command."""
    with (
        patch(
            "sam_faces.cli.list_people",
            return_value=[{"name": "Alice", "encoding_count": 2, "id": "abc"}],
        ),
        patch("sam_faces.cli.init_db"),
        patch.object(sys, "argv", ["sam-faces", "list"]),
    ):
        main()
        captured = capsys.readouterr()
        assert "Alice" in captured.out


def test_cli_unknowns(capsys):
    """Test unknowns command."""
    with (
        patch(
            "sam_faces.cli.list_unknowns",
            return_value=[
                {"id": "xyz", "detected_at": "2026-04-26", "image_path": "/tmp/p.jpg"}
            ],
        ),
        patch("sam_faces.cli.init_db"),
        patch.object(sys, "argv", ["sam-faces", "unknowns"]),
    ):
        main()
        captured = capsys.readouterr()
        assert "xyz" in captured.out
