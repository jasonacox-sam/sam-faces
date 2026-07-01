"""
Tests for sam_faces/visualize.py — bounding-box / label drawing.

visualize() delegates face detection to identify(), then draws boxes and
labels onto the image with Pillow. These tests mock identify() so they run
without face_recognition models or real photos, and exercise the drawing,
label-formatting, and output-path logic directly.

Test categories covered (per project convention):
  Security, Performance, Retry, Unit, Integration, Functional, Frame
"""

import pytest
from unittest.mock import patch
from pathlib import Path

from PIL import Image

from sam_faces.visualize import visualize

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_photo(tmp_path, name="photo.jpg", size=(200, 200), color="white"):
    """Write a small real image to disk and return its path."""
    img = Image.new("RGB", size, color)
    photo = tmp_path / name
    img.save(photo)
    return photo


def _known_face(name="Jane Smith", confidence=0.91):
    return {
        "name": name,
        "confidence": confidence,
        "unknown": False,
        "bounding_box": {"top": 40, "right": 120, "bottom": 120, "left": 40},
    }


def _unknown_face():
    return {
        "name": "Unknown",
        "confidence": None,
        "unknown": True,
        "bounding_box": {"top": 10, "right": 190, "bottom": 90, "left": 130},
    }


def _mock_result(faces, llm_context="context"):
    return {
        "face_count": len(faces),
        "faces": faces,
        "llm_context": llm_context,
    }


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


def test_security_unusual_output_filename_is_written_verbatim(tmp_path):
    """A caller-supplied output path with odd characters is honoured as a
    plain filename and never interpreted/executed — output lands exactly
    where requested and nowhere else."""
    photo = _make_photo(tmp_path)
    weird = tmp_path / "we ird;rm -rf.jpg"

    with patch(
        "sam_faces.visualize.identify",
        return_value=_mock_result([_known_face()]),
    ):
        result = visualize(str(photo), output_path=str(weird))

    assert result["output_path"] == str(weird)
    assert weird.exists()
    # No stray files created outside the intended output path.
    written = {p.name for p in tmp_path.iterdir()}
    assert written == {"photo.jpg", "we ird;rm -rf.jpg"}


def test_security_identify_error_does_not_leak_or_write(tmp_path):
    """When identify() reports an error, visualize surfaces it and writes no
    file — no partial artifacts left on disk."""
    photo = _make_photo(tmp_path)
    with patch(
        "sam_faces.visualize.identify",
        return_value={"error": "bad model path /etc/shadow"},
    ):
        result = visualize(str(photo), output_path=str(tmp_path / "out.jpg"))

    assert result["error"] == "bad model path /etc/shadow"
    assert result["output_path"] is None
    assert not (tmp_path / "out.jpg").exists()


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_performance_identify_called_exactly_once(tmp_path):
    """The (expensive) detection pass must run once per visualize call — no
    redundant re-identification."""
    photo = _make_photo(tmp_path)
    with patch(
        "sam_faces.visualize.identify",
        return_value=_mock_result([_known_face(), _unknown_face()]),
    ) as mock_identify:
        visualize(str(photo), output_path=str(tmp_path / "out.jpg"))

    assert mock_identify.call_count == 1


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="N/A: visualize() performs a single deterministic drawing pass "
    "with no network/IO retry semantics to exercise."
)
def test_retry_not_applicable():
    """Placeholder — visualize has no retry logic."""
    pass


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------


def test_unit_return_contract_for_known_face(tmp_path):
    """visualize returns the expected keys and passes identify's metadata
    straight through."""
    photo = _make_photo(tmp_path)
    faces = [_known_face()]
    with patch(
        "sam_faces.visualize.identify",
        return_value=_mock_result(faces, llm_context="one known face"),
    ):
        result = visualize(str(photo), output_path=str(tmp_path / "out.jpg"))

    assert result["face_count"] == 1
    assert result["faces"] == faces
    assert result["llm_context"] == "one known face"
    assert result["output_path"] == str(tmp_path / "out.jpg")


def test_unit_default_output_path_derivation(tmp_path):
    """With no output_path, the result is written next to the source using the
    `<stem>_faces<suffix>` naming scheme."""
    photo = _make_photo(tmp_path, name="crowd.jpg")
    with patch(
        "sam_faces.visualize.identify",
        return_value=_mock_result([_known_face()]),
    ):
        result = visualize(str(photo))

    expected = tmp_path / "crowd_faces.jpg"
    assert result["output_path"] == str(expected)
    assert expected.exists()


def test_unit_threshold_forwarded_to_identify(tmp_path):
    """The threshold argument is forwarded to identify()."""
    photo = _make_photo(tmp_path)
    with patch(
        "sam_faces.visualize.identify",
        return_value=_mock_result([_known_face()]),
    ) as mock_identify:
        visualize(str(photo), output_path=str(tmp_path / "out.jpg"), threshold=0.42)

    _, kwargs = mock_identify.call_args
    assert kwargs["threshold"] == 0.42


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def test_integration_writes_openable_annotated_image(tmp_path):
    """End-to-end with a real on-disk image (identify mocked): the annotated
    output is a valid image file that Pillow can re-open, and drawing actually
    mutated pixels relative to the blank source."""
    photo = _make_photo(tmp_path, color="white")
    out = tmp_path / "annotated.jpg"
    with patch(
        "sam_faces.visualize.identify",
        return_value=_mock_result([_known_face(), _unknown_face()]),
    ):
        result = visualize(str(photo), output_path=str(out))

    assert result["output_path"] == str(out)
    reopened = Image.open(out)
    reopened.load()
    assert reopened.size == (200, 200)
    # A blank white image should no longer be entirely white after boxes drawn.
    colors = reopened.convert("RGB").getcolors(maxcolors=100000)
    assert colors is not None
    assert len(colors) > 1


# ---------------------------------------------------------------------------
# Functional
# ---------------------------------------------------------------------------


def test_functional_mixed_known_and_unknown_faces(tmp_path):
    """The "who is this?" workflow: a group photo with one known and one
    unknown face is annotated and the counts/metadata reflect both."""
    photo = _make_photo(tmp_path)
    faces = [_known_face("Bob", 0.77), _unknown_face()]
    with patch(
        "sam_faces.visualize.identify",
        return_value=_mock_result(faces, llm_context="Bob (77%) and 1 unknown"),
    ):
        result = visualize(str(photo), output_path=str(tmp_path / "group.jpg"))

    assert result["face_count"] == 2
    names = {f["name"] for f in result["faces"]}
    assert names == {"Bob", "Unknown"}
    assert "unknown" in result["llm_context"].lower()
    assert Path(result["output_path"]).exists()


# ---------------------------------------------------------------------------
# Frame (boundary / edge conditions)
# ---------------------------------------------------------------------------


def test_frame_no_faces_returns_error(tmp_path):
    """Zero detected faces is a boundary case: return an error, write nothing."""
    photo = _make_photo(tmp_path)
    out = tmp_path / "out.jpg"
    with patch(
        "sam_faces.visualize.identify",
        return_value=_mock_result([], llm_context="No faces detected in this image."),
    ):
        result = visualize(str(photo), output_path=str(out))

    assert result["error"] == "No faces detected"
    assert result["output_path"] is None
    assert not out.exists()


def test_frame_label_above_and_below_box(tmp_path):
    """Boundary in label placement: a box near the top edge forces the label
    below the box, while a box lower in the frame draws the label above it.
    Both must render without raising."""
    photo = _make_photo(tmp_path, size=(300, 300))
    top_face = {
        "name": "Edge",
        "confidence": 0.9,
        "unknown": False,
        # top == 2, smaller than the label height -> label drawn below box
        "bounding_box": {"top": 2, "right": 100, "bottom": 80, "left": 10},
    }
    low_face = {
        "name": "Lower",
        "confidence": 0.9,
        "unknown": False,
        # plenty of room above -> label drawn above box
        "bounding_box": {"top": 150, "right": 250, "bottom": 230, "left": 160},
    }
    with patch(
        "sam_faces.visualize.identify",
        return_value=_mock_result([top_face, low_face]),
    ):
        result = visualize(str(photo), output_path=str(tmp_path / "edge.jpg"))

    assert result["face_count"] == 2
    assert Path(result["output_path"]).exists()
