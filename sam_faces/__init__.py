"""
sam-faces — Face recognition and identity memory for AI assistants.

This package provides local face recognition capabilities for OpenClaw agents:
- Enroll known people with reference photos
- Identify faces in inbound images with confidence scores
- Generate llm_context strings for enriched image descriptions
- Draw bounding boxes and labels on photos for visualization

All operations run 100% locally via dlib/face_recognition. No cloud APIs.

Schema (SQLite):
  people(id, name, created_at)
  encodings(id, person_id, vector BLOB, note, added_at, crop_path)
  unknown_candidates(id, image_path, face_crop_path, detected_at, resolved, resolved_as)

Authors:
  - Sam Cox (https://github.com/jasonacox-sam)
  - Jason Cox (https://github.com/jasonacox)
"""

__version__ = "1.0.0"
__author__ = "Sam Cox (https://github.com/jasonacox-sam), Jason Cox (https://github.com/jasonacox)"

# Lazy imports - only load heavy dependencies when needed
# Importing face_recognition at module load time triggers dlib init,
# which can fail if pkg_resources is missing (Python 3.13+ without setuptools)


def _lazy(name):
    """Lazy module loader - imports on first attribute access."""
    import importlib

    return importlib.import_module(f"sam_faces.{name}")


# Database operations are lightweight (SQLite only) - safe to import eagerly
from .database import init_db, get_all_encodings, add_person, add_encoding
from .database import find_person_by_name, list_people, add_unknown

# Heavy vision imports are deferred until actually used
identify = None
enroll = None
visualize = None
DEFAULT_THRESHOLD = 0.55


def __getattr__(name):
    """Lazy load heavy vision modules on first access."""
    global identify, enroll, visualize, DEFAULT_THRESHOLD
    if name == "identify":
        from .identify import identify as _identify, DEFAULT_THRESHOLD as _thresh

        identify = _identify
        DEFAULT_THRESHOLD = _thresh
        return identify
    if name == "enroll":
        from .enroll import enroll as _enroll

        enroll = _enroll
        return enroll
    if name == "visualize":
        from .visualize import visualize as _viz

        visualize = _viz
        return visualize
    if name == "DEFAULT_THRESHOLD":
        from .identify import DEFAULT_THRESHOLD as _thresh

        DEFAULT_THRESHOLD = _thresh
        return DEFAULT_THRESHOLD
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "init_db",
    "get_all_encodings",
    "add_person",
    "add_encoding",
    "find_person_by_name",
    "list_people",
    "add_unknown",
    "identify",
    "enroll",
    "visualize",
    "DEFAULT_THRESHOLD",
    "__version__",
]
