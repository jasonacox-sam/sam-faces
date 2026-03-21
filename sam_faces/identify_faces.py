#!/usr/bin/env python3
"""
identify_faces.py — Identify faces in a photo and return structured JSON

Usage:
    python3 identify_faces.py --photo /path/to/photo.jpg
    python3 identify_faces.py --photo photo.jpg --threshold 0.85 --save-unknowns

Output JSON:
    {
      "face_count": 2,
      "faces": [
        {
          "name": "Jason Cox",
          "confidence": 0.94,
          "unknown": false,
          "bounding_box": {"top": 120, "right": 340, "bottom": 280, "left": 180},
          "center": [260, 200],
          "position_desc": "upper-left"
        },
        {
          "name": "Unknown",
          "confidence": null,
          "unknown": true,
          "unknown_id": "abc123",
          "bounding_box": {"top": 80, "right": 600, "bottom": 240, "left": 450},
          "center": [525, 160],
          "position_desc": "upper-right"
        }
      ],
      "llm_context": "2 faces detected: Jason Cox (upper-left, 94% confidence); Unknown person (upper-right)."
    }
"""

import argparse
import json
import sys
import os
from pathlib import Path

import face_recognition
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from face_db import init_db, get_all_encodings, add_unknown

DEFAULT_THRESHOLD = 0.55  # lower distance = better match; 0.55 is a good balance

def position_desc(loc, img_h, img_w):
    top, right, bottom, left = loc
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    v = "upper" if center_y < img_h // 3 else ("middle" if center_y < 2 * img_h // 3 else "lower")
    h = "left" if center_x < img_w // 3 else ("center" if center_x < 2 * img_w // 3 else "right")
    return f"{v}-{h}" if h != "center" else v

def identify(photo_path: str, threshold: float = DEFAULT_THRESHOLD,
             save_unknowns: bool = True, save_crops: bool = True) -> dict:
    init_db()

    path = Path(photo_path)
    if not path.exists():
        return {"error": f"File not found: {photo_path}", "face_count": 0, "faces": []}

    image = face_recognition.load_image_file(str(path))
    img_h, img_w = image.shape[:2]

    locations = face_recognition.face_locations(image, model="hog")
    encodings = face_recognition.face_encodings(image, locations)

    if not encodings:
        return {
            "face_count": 0,
            "faces": [],
            "llm_context": "No faces detected in this image."
        }

    # Load known encodings from DB
    known = get_all_encodings()
    known_vectors = [e["vector"] for e in known]
    known_names = [e["name"] for e in known]

    faces = []
    for encoding, loc in zip(encodings, locations):
        top, right, bottom, left = loc
        center = [(left + right) // 2, (top + bottom) // 2]
        pos = position_desc(loc, img_h, img_w)

        if known_vectors:
            distances = face_recognition.face_distance(known_vectors, encoding)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])
            confidence = round(1.0 - best_dist, 3)

            if best_dist <= threshold:
                faces.append({
                    "name": known_names[best_idx],
                    "confidence": round(confidence, 3),
                    "unknown": False,
                    "bounding_box": {"top": top, "right": right, "bottom": bottom, "left": left},
                    "center": center,
                    "position_desc": pos
                })
                continue

        # Unknown face
        unknown_id = None
        if save_unknowns:
            crop_path = ""
            if save_crops:
                crops_dir = Path(__file__).parent.parent / "faces" / "unknown"
                crops_dir.mkdir(parents=True, exist_ok=True)
                from PIL import Image as PILImage
                img_pil = PILImage.fromarray(image)
                crop = img_pil.crop((left, top, right, bottom))
                crop_path = str(crops_dir / f"unknown_{path.stem}_{top}_{left}.jpg")
                crop.save(crop_path)
            unknown_id = add_unknown(str(path), crop_path)

        faces.append({
            "name": "Unknown",
            "confidence": None,
            "unknown": True,
            "unknown_id": unknown_id,
            "bounding_box": {"top": top, "right": right, "bottom": bottom, "left": left},
            "center": center,
            "position_desc": pos
        })

    # Build LLM context string
    parts = []
    for f in faces:
        if f["unknown"]:
            parts.append(f"Unknown person ({f['position_desc']})")
        else:
            pct = int(f["confidence"] * 100)
            parts.append(f"{f['name']} ({f['position_desc']}, {pct}% confidence)")

    llm_context = f"{len(faces)} face{'s' if len(faces) != 1 else ''} detected: " + "; ".join(parts) + "."

    return {
        "face_count": len(faces),
        "faces": faces,
        "llm_context": llm_context
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify faces in a photo")
    parser.add_argument("--photo", required=True, help="Path to photo")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Match threshold (default: {DEFAULT_THRESHOLD}; lower = stricter)")
    parser.add_argument("--no-save-unknowns", action="store_true",
                        help="Don't save unknown faces to the candidates table")
    parser.add_argument("--no-crops", action="store_true",
                        help="Don't save cropped face images for unknowns")
    args = parser.parse_args()

    result = identify(
        args.photo,
        threshold=args.threshold,
        save_unknowns=not args.no_save_unknowns,
        save_crops=not args.no_crops
    )
    print(json.dumps(result, indent=2))
