"""Pytest configuration for chinese-instrument-demucs.

Adds the vendored demucs to sys.path so imports work.
"""

import sys
from pathlib import Path

VENDOR_DEMUCS = Path(__file__).resolve().parents[1] / "vendor" / "demucs"
if str(VENDOR_DEMUCS) not in sys.path:
    sys.path.insert(0, str(VENDOR_DEMUCS))
