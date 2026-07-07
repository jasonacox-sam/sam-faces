"""
sam_faces/pets.py — recognize known pets from a photo.

Face recognition (dlib/face_recognition) only works on human faces, so pets are identified
with a local vision-language model via Ollama, matched against a small registry stored in the
same database as people (a `pets` table). Great for security-camera / doorbell setups where an
animal detection should be named ("Bailey") or flagged as an unknown animal.

Config (environment):
  SAM_FACES_VLM_URL    Ollama chat endpoint  (default: http://localhost:11434/api/chat)
  SAM_FACES_VLM_MODEL  vision model to use    (default: llava — any Ollama vision model works,
                                               e.g. llava, llama3.2-vision, qwen2.5vl, moondream)
"""
import os
import json
import base64
import urllib.request
from datetime import datetime, timezone

from .database import get_conn

VLM_URL = os.environ.get("SAM_FACES_VLM_URL", "http://localhost:11434/api/chat")
VLM_MODEL = os.environ.get("SAM_FACES_VLM_MODEL", "llava")


def _ensure_table():
    with get_conn() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pets ("
            "name TEXT PRIMARY KEY, species TEXT, description TEXT, added_at TEXT NOT NULL)"
        )


def add_pet(name: str, species: str, description: str):
    """Register (or update) a known pet and its visual description."""
    _ensure_table()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO pets (name, species, description, added_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET species=excluded.species, description=excluded.description",
            (name, species, description, datetime.now(timezone.utc).isoformat()),
        )


def list_pets() -> list[dict]:
    _ensure_table()
    with get_conn() as conn:
        rows = conn.execute("SELECT name, species, description FROM pets ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def _vlm(image_path: str, prompt: str, num_predict: int = 400) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = json.dumps({
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": num_predict},
    }).encode()
    req = urllib.request.Request(VLM_URL, data=payload, headers={"Content-Type": "application/json"})
    data = json.loads(urllib.request.urlopen(req, timeout=60).read())
    content = data.get("message", {}).get("content", "") or ""
    if "<think>" in content:                 # some reasoning models wrap their answer
        end = content.rfind("</think>")
        content = content[end + 8:] if end > 0 else content
    return content.strip()


def describe(image_path: str) -> str:
    """One-sentence description of the animal in an image — handy for seeding the registry."""
    return _vlm(image_path, "Describe ONLY the animal in this image in one sentence: species, "
                "size, coat color/pattern, and any distinctive features. No preamble.")


def _match(answer: str, pets: list[dict]) -> str | None:
    """Map a model answer to a registered pet name (substring, case-insensitive)."""
    for p in pets:
        if p["name"].lower() in (answer or "").lower():
            return p["name"]
    return None


def identify_pet(image_path: str) -> str | None:
    """Return a known pet's name, or None for an unknown animal / no animal.

    None is the alert-worthy case for a camera setup: an animal that is not one of yours."""
    pets = list_pets()
    if not pets:
        return None
    roster = "\n".join(f"- {p['name']}: {p['description']}" for p in pets)
    prompt = ("Identify the pet in this photo.\n"
              f"Known pets:\n{roster}\n\n"
              "If the animal clearly matches ONE known pet, reply with ONLY that name. "
              "If it's an animal that matches none, reply 'UNKNOWN_ANIMAL'. "
              "If there's no animal, reply 'NONE'.")
    return _match(_vlm(image_path, prompt).strip('".\''), pets)
