#!/usr/bin/env python3
"""
visualize.py — Generate annotated face recognition demo images for sam-faces.

Usage:
    python3 visualize.py --photo input.jpg --output annotated.jpg [--db path/to/people.db]
"""

import argparse
import os
import sqlite3
import numpy as np
import face_recognition
from PIL import Image, ImageDraw, ImageFont

DEFAULT_DB = os.environ.get("SAM_FACES_DB", "faces/people.db")
THRESHOLD = 0.55

COLORS = [
    (52, 211, 153),   # green
    (96, 165, 250),   # blue
    (251, 191, 36),   # yellow
    (248, 113, 113),  # red
    (167, 139, 250),  # purple
    (251, 146, 60),   # orange
]


def load_db_encodings(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT p.name, e.vector
        FROM encodings e
        JOIN people p ON e.person_id = p.id
    """)
    rows = c.fetchall()
    conn.close()
    known = {}
    for name, vec_bytes in rows:
        vec = np.frombuffer(vec_bytes, dtype=np.float64)
        known.setdefault(name, []).append(vec)
    return known


def identify(face_enc, known):
    best_name, best_conf = "Unknown", None
    best_dist = 1.0
    for name, encs in known.items():
        dists = face_recognition.face_distance(encs, face_enc)
        d = float(np.min(dists))
        if d < best_dist:
            best_dist = d
            best_name = name
            best_conf = round(1.0 - d, 3)
    if best_dist > THRESHOLD:
        return "Unknown", None
    return best_name, best_conf


def annotate(photo_path, output_path, db_path=DEFAULT_DB):
    known = load_db_encodings(db_path)
    img = face_recognition.load_image_file(photo_path)
    locations = face_recognition.face_locations(img, model="hog")
    encodings = face_recognition.face_encodings(img, locations)

    pil = Image.open(photo_path).convert("RGB")
    draw = ImageDraw.Draw(pil)

    # Try to load a font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    results = []
    for i, (loc, enc) in enumerate(zip(locations, encodings)):
        top, right, bottom, left = loc
        name, conf = identify(enc, known)
        color = COLORS[i % len(COLORS)]

        # Draw bounding box (thick)
        for t in range(3):
            draw.rectangle([(left - t, top - t), (right + t, bottom + t)], outline=color)

        # Label background
        label = f"{name}" if name == "Unknown" else f"{name}  {int(conf*100)}%"
        bbox = font.getbbox(label)
        lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 4
        label_y = top - lh - pad * 2 - 2
        if label_y < 0:
            label_y = bottom + 2
        draw.rectangle(
            [(left, label_y), (left + lw + pad * 2, label_y + lh + pad * 2)],
            fill=color
        )
        draw.text((left + pad, label_y + pad), label, fill=(0, 0, 0), font=font)

        results.append({"name": name, "confidence": conf, "box": loc})
        print(f"  [{i}] {name} — {int(conf*100)}% confidence" if conf else f"  [{i}] Unknown")

    pil.save(output_path, quality=92)
    print(f"✅ Saved annotated image: {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate faces in a photo")
    parser.add_argument("--photo", required=True, help="Input photo path")
    parser.add_argument("--output", required=True, help="Output annotated image path")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to people.db")
    args = parser.parse_args()
    annotate(args.photo, args.output, args.db)
