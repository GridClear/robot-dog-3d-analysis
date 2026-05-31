"""Session directories and artifact paths."""
from __future__ import annotations

import secrets
import shutil
from pathlib import Path

from app.config import get_settings

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def new_job_id() -> str:
    return secrets.token_hex(6)


def session_dir(job_id: str) -> Path:
    return get_settings().sessions_dir / job_id


def inputs_dir(job_id: str) -> Path:
    return session_dir(job_id) / "inputs"


def frames_dir(job_id: str) -> Path:
    return session_dir(job_id) / "frames"


def colmap_dir(job_id: str) -> Path:
    return session_dir(job_id) / "colmap"


def train_dir(job_id: str) -> Path:
    return session_dir(job_id) / "train"


def artifacts_dir(job_id: str) -> Path:
    return session_dir(job_id) / "artifacts"


def logs_dir(job_id: str) -> Path:
    return session_dir(job_id) / "logs"


def save_frames(job_id: str, images: list[tuple[str, bytes]]) -> list[Path]:
    """Save uploaded image bytes; return ordered frame paths."""
    fdir = frames_dir(job_id)
    fdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, (name, data) in enumerate(images):
        suffix = Path(name).suffix.lower()
        if suffix not in _IMG_SUFFIXES:
            suffix = ".jpg"
        p = fdir / f"{i:04d}{suffix}"
        p.write_bytes(data)
        paths.append(p)
    return paths


def save_video(job_id: str, filename: str, data: bytes) -> Path:
    idir = inputs_dir(job_id)
    idir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    if suffix not in _VIDEO_SUFFIXES:
        suffix = ".mp4"
    path = idir / f"source{suffix}"
    path.write_bytes(data)
    return path


def frame_paths(job_id: str) -> list[Path]:
    fdir = frames_dir(job_id)
    if not fdir.exists():
        return []
    return sorted(p for p in fdir.iterdir() if p.suffix.lower() in _IMG_SUFFIXES)


def artifact_path(job_id: str, name: str) -> Path | None:
    mapping = {
        "scene.ply": artifacts_dir(job_id) / "scene.ply",
        "scene.splat": artifacts_dir(job_id) / "scene.splat",
        "cameras.json": artifacts_dir(job_id) / "cameras.json",
        "preview.jpg": artifacts_dir(job_id) / "preview.jpg",
    }
    p = mapping.get(name)
    return p if p and p.exists() else None


def cleanup_old_sessions(max_age_hours: int) -> int:
    """Remove session dirs older than max_age_hours. Returns count removed."""
    import time

    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    root = get_settings().sessions_dir
    if not root.exists():
        return 0
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed
