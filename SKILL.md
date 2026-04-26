---
name: sam-faces
description: >
  Face recognition and identity memory for AI assistants. Enroll known people
  with reference photos, then automatically identify faces in inbound images —
  with names, confidence scores, and an llm_context string ready to inject into
  any LLM prompt. 100% local, no cloud APIs.
homepage: https://github.com/jasonacox-sam/sam-faces
metadata:
  {
    "openclaw":
      {
        "emoji": "👤",
        "requires": { "bins": ["python3"] },
        "install":
          [
            {
              "id": "pip",
              "kind": "pip",
              "package": "sam-faces",
              "bins": ["sam-faces"],
              "label": "Install via PyPI",
            },
          ],
      },
  }
---

# sam-faces

Give your OpenClaw agent a real face memory. Enroll people with reference
photos, then automatically identify them in inbound images.

## Automatic behavior (always on)

**When the user sends a photo or image:**

1. Save the image to a temp path.
2. Run: `sam-faces identify <path>`
3. Parse the JSON response.
4. If `face_count > 0`, prepend `llm_context` to your image description.
5. If any face has `"unknown": true`, ask the user who they are, then enroll them.

Do this automatically — do not ask permission first.

## Installation

### Via PyPI (recommended)

```bash
pip install sam-faces
```

The `sam-faces` CLI is added to your PATH automatically.

### Manual (from source)

```bash
git clone https://github.com/jasonacox-sam/sam-faces.git
cd sam-faces
pip install -e .
```

### Via OpenClaw Skills UI

Search "face recognition" in the OpenClaw Skills UI and install with one click.

## CLI Usage

### Identify faces in a photo

```bash
sam-faces identify /path/to/photo.jpg
```

Returns JSON with names, confidence scores, bounding boxes, position, and an
`llm_context` string.

### Enroll a person

```bash
sam-faces enroll --name "Jane Smith" --photo /path/to/photo.jpg
```

If multiple faces are detected, specify which to enroll:

```bash
sam-faces enroll --name "Jane Smith" --photo photo.jpg --face-index 1
```

One reference photo is usually enough (default threshold: 0.55).
Enroll 2–3 photos across different lighting for best accuracy.

### List enrolled people

```bash
sam-faces list
```

### Review unknown faces

```bash
sam-faces unknowns
```

## Thresholds

- Default: `--threshold 0.55` (good balance of precision and recall)
- Stricter: `--threshold 0.45` — fewer false positives
- Looser: `--threshold 0.65` — better recall in varied lighting

## Notes

- All inference runs locally via `face_recognition` (dlib). Nothing leaves the machine.
- Database: `{workspaceDir}/faces/people.db`
- Unknown face crops: `{workspaceDir}/faces/unknown/`
- Face crop thumbnails (audit trail): `{workspaceDir}/faces/crops/`
- Requires Python 3.9+ and build tools (cmake, C++ compiler) for dlib.
  - On Ubuntu/Debian: `sudo apt install cmake build-essential`
  - On macOS: `xcode-select --install`
