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
        "requires": { "bins": ["sam-faces"] },
      },
  }
---

# sam-faces

Give your OpenClaw agent a real face memory. Enroll people with reference
photos, then automatically identify them in inbound images.

## Installation

Clone this repo into your OpenClaw workspace skills directory, then run setup:

```bash
git clone https://github.com/jasonacox-sam/sam-faces.git ~/.openclaw/workspace/skills/sam-faces
bash ~/.openclaw/workspace/skills/sam-faces/scripts/setup.sh
```

Setup installs `face_recognition`, `Pillow`, and `numpy` via pip, then adds a
`sam-faces` command to `~/.local/bin`. Start a new OpenClaw session and the
skill will be active.

## Automatic behavior (always on)

**When the user sends a photo or image:**

1. Save the image to a temp path.
2. Run: `sam-faces --photo <path>`
3. Parse the JSON response.
4. If `face_count > 0`, prepend `llm_context` to your image description.
5. If any face has `"unknown": true`, ask the user who they are, then enroll them.

Do this automatically — do not ask permission first.

## Enroll a person

```bash
python3 {baseDir}/sam_faces/enroll_face.py \
  --name "Jane Smith" \
  --photo /path/to/photo.jpg
```

- One reference photo is usually enough (default threshold: 0.55).
- Enroll 2–3 photos across different lighting for best accuracy.
- Encodings are stored in `{workspaceDir}/faces/people.db`.

## Identify faces

```bash
sam-faces --photo /path/to/image.jpg
```

Returns JSON with names, confidence scores, bounding boxes, and an
`llm_context` string:

```json
{
  "face_count": 2,
  "faces": [
    {
      "name": "Jane Smith",
      "confidence": 0.646,
      "unknown": false,
      "position_desc": "middle"
    }
  ],
  "llm_context": "2 faces detected: Jane Smith (middle, 64%); John Smith (left, 57%)."
}
```

## Thresholds

- Default: `--threshold 0.55` (good balance of precision and recall)
- Stricter: `--threshold 0.45` — fewer false positives
- Looser: `--threshold 0.65` — better recall in varied lighting

## Notes

- All inference runs locally via `face_recognition` (dlib). Nothing leaves the machine.
- Database: `{workspaceDir}/faces/people.db`
- Unknown face crops: `{workspaceDir}/faces/unknown/`
- Requires Python 3.8+ and build tools (cmake, C++ compiler) for dlib.
  On Ubuntu/Debian: `sudo apt install cmake build-essential`
  On macOS: `xcode-select --install`
