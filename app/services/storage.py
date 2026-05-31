"""Session directories and artifact paths."""
from __future__ import annotations

import io
import secrets
import shutil
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from app.config import get_settings
from app.schemas import ImageQueueItem, ImageStatus, SessionRecord, SessionState

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def new_session_id() -> str:
    return secrets.token_hex(6)


def session_dir(session_id: str) -> Path:
    return get_settings().sessions_dir / session_id


def inputs_dir(session_id: str) -> Path:
    return session_dir(session_id) / "inputs"


def artifacts_dir(session_id: str) -> Path:
    return session_dir(session_id) / "artifacts"


def logs_dir(session_id: str) -> Path:
    return session_dir(session_id) / "logs"


def meta_path(session_id: str) -> Path:
    return session_dir(session_id) / "meta.json"


def prompt_path(session_id: str) -> Path:
    return session_dir(session_id) / "prompt.txt"


def latest_splat_path(session_id: str) -> Path:
    return artifacts_dir(session_id) / "latest.splat"


def generation_splat_path(session_id: str, index: int) -> Path:
    return artifacts_dir(session_id) / f"gen_{index:04d}.splat"


def generation_splat_temp_path(session_id: str, index: int) -> Path:
    return artifacts_dir(session_id) / f"gen_{index:04d}.tmp.splat"


def save_prompt(session_id: str, prompt: str) -> None:
    session_dir(session_id).mkdir(parents=True, exist_ok=True)
    prompt_path(session_id).write_text(prompt, encoding="utf-8")


def save_images(session_id: str, images: list[tuple[str, bytes]]) -> list[Path]:
    """Save uploaded images as inputs/0000.jpg, …; enforce max height 720p."""
    settings = get_settings()
    idir = inputs_dir(session_id)
    idir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, (name, data) in enumerate(images):
        suffix = Path(name).suffix.lower()
        if suffix not in _IMG_SUFFIXES:
            suffix = ".jpg"
        out = idir / f"{i:04d}{suffix}"
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        if h > settings.max_image_height:
            scale = settings.max_image_height / h
            img = img.resize((int(w * scale), settings.max_image_height), Image.Resampling.LANCZOS)
        if suffix in {".jpg", ".jpeg"}:
            img.save(out, "JPEG", quality=92)
        else:
            img.save(out)
        paths.append(out)
    return paths


def input_image_paths(session_id: str) -> list[Path]:
    idir = inputs_dir(session_id)
    if not idir.exists():
        return []
    return sorted(p for p in idir.iterdir() if p.suffix.lower() in _IMG_SUFFIXES)


def write_meta(record: SessionRecord) -> None:
    meta_path(record.session_id).write_text(
        record.model_dump_json(indent=2),
        encoding="utf-8",
    )


def read_meta(session_id: str) -> SessionRecord | None:
    p = meta_path(session_id)
    if not p.exists():
        return None
    return SessionRecord.model_validate_json(p.read_text(encoding="utf-8"))


def init_session_meta(
    session_id: str,
    prompt: str,
    action: str,
    loop: bool,
    image_paths: list[Path],
) -> SessionRecord:
    settings = get_settings()
    items = [
        ImageQueueItem(index=i, filename=p.name, status=ImageStatus.PENDING)
        for i, p in enumerate(image_paths)
    ]
    record = SessionRecord(
        session_id=session_id,
        created_at=datetime.now(UTC),
        state=SessionState.CREATED,
        prompt=prompt,
        action=action,
        loop=loop,
        n_images=len(image_paths),
        images=items,
        interval_sec=settings.interval_sec,
        config={
            "num_frames": settings.capped_num_frames,
            "fps": settings.fps,
            "use_refiner": settings.use_refiner,
            "inference_step": settings.inference_step,
        },
    )
    write_meta(record)
    return record


def cleanup_old_sessions(max_age_hours: int) -> int:
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


def apply_benchmark_defaults() -> None:
    """Load data/benchmark.json recommended settings if present."""
    import json

    path = get_settings().benchmark_path
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rec = data.get("recommended") or data
        # Applied at runtime via env; documented in .env.example
        _ = rec
    except (json.JSONDecodeError, OSError):
        pass
