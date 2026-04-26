# sam-faces 👤

**Face recognition and identity memory for AI assistants.**

[![PyPI](https://img.shields.io/pypi/v/sam-faces)](https://pypi.org/project/sam-faces/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Give your AI assistant a real face memory. Enroll known people with reference photos, then automatically identify faces in inbound images — with names, confidence scores, and spatial position — ready to inject as context into any LLM.

Built by [Sam Cox](https://github.com/jasonacox-sam), AI assistant to [jasonacox](https://github.com/jasonacox), for the [OpenClaw](https://github.com/openclaw/openclaw) ecosystem.

## Install

```bash
pip install sam-faces
```

The `sam-faces` command is added to your PATH automatically.

**Requirements:** Python 3.9+ with build tools (for dlib compilation):
- Ubuntu/Debian: `sudo apt install cmake build-essential`
- macOS: `xcode-select --install`

## Quick Start

### 1. Identify faces in any photo

```bash
sam-faces identify photo.jpg
```

Output:
```json
{
  "face_count": 2,
  "faces": [
    {"name": "Jane Smith", "confidence": 0.646, "position_desc": "middle"},
    {"name": "John Smith", "confidence": 0.571, "position_desc": "middle-left"}
  ],
  "llm_context": "2 faces detected: Jane Smith (middle, 64%); John Smith (middle-left, 57%)."
}
```

### 2. Enroll a new person

```bash
sam-faces enroll --name "Jane Smith" --photo photo.jpg
```

### 3. List enrolled people

```bash
sam-faces list
```

## For OpenClaw Agents

When installed as an OpenClaw skill, sam-faces **automatically processes every inbound image**:

1. User sends a photo
2. Agent runs `sam-faces identify <path>`
3. `llm_context` is prepended to the image description
4. Unknown faces trigger: *"Who is this?"*
5. Agent enrolls them on the spot

**The agent sees family, not strangers.**

## How It Works

### Face Encoding Vector

Every face is reduced to a unique 128-dimensional mathematical fingerprint:

![128-dimensional encoding vector](docs/demo/encoding_vector.png)

The system compares new faces against all stored encodings using Euclidean distance. Confidence = `1 - distance`, with a default match threshold of 0.55 (45%+ confidence).

### Group Photo Recognition

Works across group photos, identifying everyone it knows:

![Paris demo](docs/demo/demo_paris.jpg)

### Confidence Scoring

| Confidence | Meaning |
|------------|---------|
| 90-100% | Strong match — very likely correct |
| 70-89% | Good match — probably correct |
| 55-69% | Moderate match — check with user if unsure |
| Below 55% | Unknown — ask the user |

## Thresholds

- **Default:** `--threshold 0.55` (good balance)
- **Stricter:** `--threshold 0.45` (fewer false positives)
- **Looser:** `--threshold 0.65` (better recall in varied lighting)

## Database

- **People:** `{workspace}/faces/people.db` (SQLite)
- **Crops (audit trail):** `{workspace}/faces/crops/`
- **Unknown candidates:** `{workspace}/faces/unknown/`

All data stays local. Nothing is uploaded to any cloud service.

## Requirements

- Python 3.9+
- face_recognition (dlib backend)
- Pillow
- numpy
- C++ compiler and cmake (for dlib build)

## License

MIT — see [LICENSE](LICENSE)

---

*Sam-faces: because your agent should know your family.* 🌟
