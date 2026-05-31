"""Video frame extraction via ffmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path

from app.config import get_settings


def extract_frames(video_path: Path, out_dir: Path, fps: int | None = None) -> list[Path]:
    settings = get_settings()
    out_dir.mkdir(parents=True, exist_ok=True)
    rate = fps if fps is not None else settings.video_fps
    pattern = str(out_dir / "%04d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={rate}",
        pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-2000:]}")

    frames = sorted(out_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError("ffmpeg produced no frames")
    return frames


def dedupe_frames(frames: list[Path], max_keep: int | None = None) -> list[Path]:
    """Drop near-duplicate consecutive frames using perceptual hash."""
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        return frames[:max_keep] if max_keep else frames

    kept: list[Path] = []
    last_hash = None
    for p in frames:
        h = imagehash.phash(Image.open(p))
        if last_hash is not None and h - last_hash < 2:
            continue
        kept.append(p)
        last_hash = h
        if max_keep and len(kept) >= max_keep:
            break
    return kept if kept else frames[:1]
