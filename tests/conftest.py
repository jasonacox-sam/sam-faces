import pytest
import sys
from pathlib import Path

# Ensure sam_faces is importable during tests
sys.path.insert(0, str(Path(__file__).parent.parent))
