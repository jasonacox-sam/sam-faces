"""
sam_faces/visualize.py — Draw bounding boxes and labels on detected faces.

Green boxes for known faces (name + confidence). Red boxes for unknown faces.
Useful for the "who is this?" workflow — shows exactly which faces in a
group photo couldn't be identified.

Usage: sam-faces visualize photo.jpg --output result.jpg
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json

from .identify import identify


def visualize(photo_path: str, output_path: str = None, threshold: float = 0.55):
    """Draw bounding boxes and labels on detected faces."""
    result = identify(
        photo_path, threshold=threshold, save_unknowns=False, save_crops=False
    )

    if result.get("error"):
        return {"error": result["error"], "output_path": None}

    if result["face_count"] == 0:
        return {"error": "No faces detected", "output_path": None}

    img = Image.open(photo_path)
    draw = ImageDraw.Draw(img)

    # Try to get a font, fall back to default if not available
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
        small_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12
        )
    except:
        font = ImageFont.load_default()
        small_font = font

    for face in result["faces"]:
        bb = face["bounding_box"]
        top, right, bottom, left = bb["top"], bb["right"], bb["bottom"], bb["left"]

        if face["unknown"]:
            color = "red"
            label = "Unknown"
            conf_text = ""
        else:
            color = "green"
            label = face["name"]
            conf_text = f"{int(face['confidence'] * 100)}%"

        # Draw bounding box
        draw.rectangle([left, top, right, bottom], outline=color, width=3)

        # Draw label background
        text = f"{label} {conf_text}".strip()
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        label_y = top - text_h - 4 if top > text_h + 4 else bottom + 4
        draw.rectangle(
            [left, label_y, left + text_w + 8, label_y + text_h + 4], fill=color
        )

        # Draw label text
        draw.text((left + 4, label_y + 2), text, fill="white", font=font)

    # Save output
    if output_path:
        out = Path(output_path)
    else:
        p = Path(photo_path)
        out = p.parent / f"{p.stem}_faces{p.suffix}"

    img.save(out)

    return {
        "face_count": result["face_count"],
        "faces": result["faces"],
        "output_path": str(out),
        "llm_context": result["llm_context"],
    }
