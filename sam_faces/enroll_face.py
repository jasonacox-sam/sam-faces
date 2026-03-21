#!/usr/bin/env python3
"""
enroll_face.py — Enroll a face into Sam's people database

Usage:
    python3 enroll_face.py --name "Jason Cox" --photo /path/to/photo.jpg
    python3 enroll_face.py --name "Jason Cox" --photo photo.jpg --note "GTC 2026"

If multiple faces are detected, prints a description of each and asks which to enroll.
"""

import argparse
import sys
import face_recognition
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from face_db import init_db, add_person, add_encoding, find_person_by_name, get_all_encodings
import numpy as np

def enroll(name: str, photo_path: str, note: str = "", face_index: int = None):
    init_db()

    path = Path(photo_path)
    if not path.exists():
        print(f"❌ File not found: {photo_path}")
        sys.exit(1)

    print(f"🔍 Loading image: {path.name}")
    image = face_recognition.load_image_file(str(path))
    locations = face_recognition.face_locations(image, model="hog")
    encodings = face_recognition.face_encodings(image, locations)

    if not encodings:
        print("❌ No faces detected in this photo.")
        sys.exit(1)

    if len(encodings) == 1:
        encoding = encodings[0]
        loc = locations[0]
        print(f"✅ Found 1 face at position top={loc[0]}, right={loc[1]}, bottom={loc[2]}, left={loc[3]}")
    else:
        print(f"⚠️  Found {len(encodings)} faces in this photo:")
        for i, loc in enumerate(locations):
            # Estimate face center for description
            center_x = (loc[3] + loc[1]) // 2
            center_y = (loc[0] + loc[2]) // 2
            h, w = image.shape[:2]
            h_pos = "top" if center_y < h//3 else ("middle" if center_y < 2*h//3 else "bottom")
            v_pos = "left" if center_x < w//3 else ("center" if center_x < 2*w//3 else "right")
            print(f"  [{i}] Face at {h_pos}-{v_pos} of photo (top={loc[0]}, right={loc[1]}, bottom={loc[2]}, left={loc[3]})")

        if face_index is None:
            choice = input(f"\nWhich face is {name}? Enter number [0-{len(encodings)-1}]: ").strip()
            face_index = int(choice)

        encoding = encodings[face_index]
        loc = locations[face_index]
        print(f"✅ Using face [{face_index}] at top={loc[0]}, right={loc[1]}, bottom={loc[2]}, left={loc[3]}")

    # Check for existing person
    existing = find_person_by_name(name)
    if existing:
        print(f"📁 Found existing entry for '{name}' (id: {existing['id']}) — adding new encoding")
        person_id = existing["id"]
    else:
        person_id = add_person(name)
        print(f"👤 Created new person: '{name}' (id: {person_id})")

    # Check similarity to existing encodings for this person (avoid duplicates)
    all_encodings = get_all_encodings()
    person_encodings = [e for e in all_encodings if e["person_id"] == person_id]
    if person_encodings:
        distances = face_recognition.face_distance(
            [e["vector"] for e in person_encodings], encoding
        )
        min_dist = min(distances)
        if min_dist < 0.35:
            print(f"⚠️  Very similar encoding already exists (distance: {min_dist:.3f}). Still enrolling.")

    eid = add_encoding(person_id, encoding, note=note or path.name)
    print(f"✅ Enrolled encoding (id: {eid}) for '{name}'" + (f" — note: {note}" if note else ""))
    print(f"   Vector dimensions: {len(encoding)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a face into Sam's people database")
    parser.add_argument("--name", required=True, help="Person's name")
    parser.add_argument("--photo", required=True, help="Path to photo")
    parser.add_argument("--note", default="", help="Optional note (e.g. 'GTC 2026 headshot')")
    parser.add_argument("--face-index", type=int, default=None,
                        help="If multiple faces, which index to enroll (0-based)")
    args = parser.parse_args()

    enroll(args.name, args.photo, args.note, args.face_index)
