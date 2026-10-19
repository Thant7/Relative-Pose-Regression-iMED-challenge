"""Resolve external source checkouts without bundling their code."""
import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
def setup_external_paths():
    reloc = Path(os.environ.get("RELOC3R_ROOT", PROJECT_ROOT / "third_party" / "reloc3r"))
    roma = Path(os.environ.get("ROMAV2_ROOT", PROJECT_ROOT / "third_party" / "RoMaV2"))
    for path in (reloc / "croco", reloc, roma / "src"):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
