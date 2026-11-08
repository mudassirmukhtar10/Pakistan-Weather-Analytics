import sys
from pathlib import Path

# Add project root and src to sys.path for test discovery
ROOT_DIR = Path(__file__).parent.resolve()
SRC_DIR = ROOT_DIR / "src"

for p in [ROOT_DIR, SRC_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

