"""
sam_faces/cli.py — Unified command-line interface for sam-faces.

Subcommands:
  identify    — Detect and name faces in a photo
  enroll      — Add a face to the people database
  list        — Show all enrolled people
  unknowns    — Review unresolved unknown faces
  visualize   — Draw bounding boxes on detected faces

Backward compatibility: sam-faces --photo image.jpg routes to identify.
"""

import argparse
import json
import sys
from pathlib import Path

from .database import init_db, list_people, list_unknowns
from .identify import identify, DEFAULT_THRESHOLD
from .enroll import enroll


from .visualize import visualize


def main():
    parser = argparse.ArgumentParser(
        prog="sam-faces",
        description="Face recognition and identity memory for AI assistants",
    )
    parser.add_argument(
        "--photo",
        dest="legacy_photo",
        help="Legacy: identify faces in a photo (backward compat)",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # identify
    p_identify = sub.add_parser("identify", help="Identify faces in a photo")
    p_identify.add_argument("photo", nargs="?", help="Path to photo")
    p_identify.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Match threshold (default: {DEFAULT_THRESHOLD})",
    )
    p_identify.add_argument(
        "--no-save-unknowns", action="store_true", help="Don't save unknown faces"
    )
    p_identify.add_argument(
        "--no-crops",
        action="store_true",
        help="Don't save cropped face images for unknowns",
    )

    # enroll
    p_enroll = sub.add_parser("enroll", help="Enroll a face into the database")
    p_enroll.add_argument("--name", required=True, help="Person's name")
    p_enroll.add_argument("--photo", required=True, help="Path to photo")
    p_enroll.add_argument("--note", default="", help="Optional note")
    p_enroll.add_argument(
        "--face-index",
        type=int,
        default=None,
        help="Which face to enroll if multiple (0-based)",
    )

    # list
    p_list = sub.add_parser("list", help="List enrolled people")

    # unknowns
    p_unknowns = sub.add_parser("unknowns", help="List unresolved unknown faces")

    # visualize
    p_viz = sub.add_parser("visualize", help="Draw bounding boxes on detected faces")
    p_viz.add_argument("photo", help="Path to photo")
    p_viz.add_argument(
        "--output", "-o", help="Output path (default: <photo>_faces.jpg)"
    )
    p_viz.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Match threshold (default: {DEFAULT_THRESHOLD})",
    )

    args = parser.parse_args()

    # Handle legacy --photo flag
    if args.legacy_photo and not args.command:
        args.command = "identify"
        args.photo = args.legacy_photo
        args.threshold = DEFAULT_THRESHOLD
        args.no_save_unknowns = False
        args.no_crops = False

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command in ("identify", "--photo"):
        result = identify(
            args.photo,
            threshold=args.threshold,
            save_unknowns=not args.no_save_unknowns,
            save_crops=not args.no_crops,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "enroll":
        try:
            result = enroll(args.name, args.photo, args.note, args.face_index)
            print(json.dumps(result, indent=2))
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "list":
        init_db()
        people = list_people()
        if not people:
            print("No people enrolled yet.")
        for p in people:
            print(f"  {p['name']:30s} — {p['encoding_count']} encoding(s)  [{p['id']}]")

    elif args.command == "unknowns":
        init_db()
        unknowns = list_unknowns()
        if not unknowns:
            print("No unresolved unknown faces.")
        for u in unknowns:
            print(f"  [{u['id']}] {u['detected_at'][:10]}  {u['image_path']}")

    elif args.command == "visualize":
        result = visualize(args.photo, args.output, args.threshold)
        if result.get("error"):
            print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
