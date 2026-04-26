import tempfile
import os
from pathlib import Path
from PIL import Image

import face_recognition
import numpy as np

from .database import init_db, add_person, add_encoding, find_person_by_name, get_all_encodings

CROPS_DIR = Path.home() / ".openclaw" / "workspace" / "faces" / "crops"


def enroll(name: str, photo_path: str, note: str = "", face_index: int = None):
    """Enroll a face into the people database."""
    init_db()

    path = Path(photo_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {photo_path}")

    image = face_recognition.load_image_file(str(path))
    locations = face_recognition.face_locations(image, model="hog")
    encodings = face_recognition.face_encodings(image, locations)

    if not encodings:
        raise ValueError("No faces detected in this photo.")

    if len(encodings) == 1:
        encoding = encodings[0]
        loc = locations[0]
    else:
        if face_index is None:
            raise ValueError(
                f"Multiple faces found ({len(encodings)}). "
                f"Specify face_index (0-{len(encodings)-1})."
            )
        encoding = encodings[face_index]
        loc = locations[face_index]

    # Check for existing person
    existing = find_person_by_name(name)
    if existing:
        person_id = existing["id"]
    else:
        person_id = add_person(name)

    # Save face crop thumbnail for audit trail
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    pil_img = Image.fromarray(image)
    top, right, bottom, left = loc
    padding = 20
    crop_box = (
        max(0, left - padding),
        max(0, top - padding),
        min(pil_img.width, right + padding),
        min(pil_img.height, bottom + padding)
    )
    crop = pil_img.crop(crop_box)
    crop.thumbnail((200, 200))

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=str(CROPS_DIR))
    crop.save(tmp.name, "JPEG", quality=85)
    tmp.close()

    eid = add_encoding(person_id, encoding, note=note or path.name, crop_path=tmp.name)

    # Rename to final path using real encoding ID
    final_crop = CROPS_DIR / f"{eid}.jpg"
    os.rename(tmp.name, str(final_crop))

    # Update crop_path in DB to final filename
    import sqlite3
    from .database import get_conn
    with get_conn() as conn:
        conn.execute("UPDATE encodings SET crop_path=? WHERE id=?", (str(final_crop), eid))

    return {
        "encoding_id": eid,
        "person_id": person_id,
        "person_name": name,
        "crop_path": str(final_crop),
        "note": note or path.name
    }
