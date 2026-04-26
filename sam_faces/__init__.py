#!/usr/bin/env python3
"""
sam_faces/__init__.py — Face recognition and identity memory for AI assistants.

Enroll known people with reference photos, then automatically identify faces
in inbound images — with names, confidence scores, and an llm_context string
ready to inject into any LLM prompt. 100% local, no cloud APIs.
"""

__version__ = "1.0.0"
__author__ = "Sam Cox <sam@jasonacox.com>"

from .database import init_db, get_all_encodings, add_person, add_encoding
from .database import find_person_by_name, list_people, add_unknown
from .identify import identify, DEFAULT_THRESHOLD
from .enroll import enroll

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
    "DEFAULT_THRESHOLD",
    "__version__",
]