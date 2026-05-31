"""Live GPU tests. Skipped unless SPLAT_LIVE=1."""
from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

pytestmark = pytest.mark.skipif(
    os.environ.get("SPLAT_LIVE") != "1",
    reason="set SPLAT_LIVE=1 for GPU tests",
)


def _synthetic_frames(tmp: Path, n: int = 3) -> list[Path]:
    paths = []
    for i, dx in enumerate(range(0, n * 20, 20)):
        img = Image.new("RGB", (480, 360), (70, 70, 75))
        d = ImageDraw.Draw(img)
        d.rectangle([120 - dx, 150, 240 - dx, 220], fill=(200, 50, 50))
        d.ellipse([60 - dx, 60, 110 - dx, 110], fill=(230, 220, 60))
        p = tmp / f"view_{i}.jpg"
        img.save(p, "JPEG")
        paths.append(p)
    return paths


@pytest.mark.live
def test_vggt_pose(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_DATA_DIR", str(tmp_path / "data"))
    from app.config import clear_settings_cache

    clear_settings_cache()
    from app.services.pose.vggt import VGGTPoseBackend

    frames = _synthetic_frames(tmp_path)
    r = VGGTPoseBackend().estimate(frames, tmp_path / "work")
    assert r.available, r.note
    assert r.colmap_dir is not None
    assert (r.colmap_dir / "images").exists()
    sparse = r.colmap_dir / "sparse" / "0"
    assert sparse.exists()
