#!/usr/bin/env bash
# sam-faces setup — install dependencies and the sam-faces CLI
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$HOME/.local/bin"
BIN_PATH="$BIN_DIR/sam-faces"

echo "Installing sam-faces dependencies..."
pip install face_recognition Pillow numpy --quiet

echo "Installing sam-faces CLI to $BIN_PATH..."
mkdir -p "$BIN_DIR"
cat > "$BIN_PATH" << WRAPPER
#!/usr/bin/env bash
# sam-faces — identify faces in a photo
exec python3 "$SKILL_DIR/sam_faces/identify_faces.py" "\$@"
WRAPPER
chmod +x "$BIN_PATH"

# Remind user to add ~/.local/bin to PATH if needed
if ! echo "$PATH" | grep -q "$BIN_DIR"; then
  echo ""
  echo "Note: Add $BIN_DIR to your PATH if not already set:"
  echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc && source ~/.bashrc"
fi

echo ""
echo "✅ sam-faces ready!"
echo ""
echo "Enroll someone:  python3 $SKILL_DIR/sam_faces/enroll_face.py --name 'Name' --photo photo.jpg"
echo "Identify faces:  sam-faces --photo image.jpg"
echo ""
echo "Start a new OpenClaw session to activate the skill."
