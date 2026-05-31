"""Copy trainer Gaussian PLY to artifacts."""
from __future__ import annotations

import shutil
from pathlib import Path


def export_ply(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst
