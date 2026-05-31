"""Liveness and SANA-WM readiness."""
from __future__ import annotations

import shutil
import subprocess

from fastapi import APIRouter

from app.config import get_settings
from app.services import sana_runner

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict:
    settings = get_settings()
    sana_ok, sana_note = sana_runner.sana_ready()
    gpu: dict = {"available": False}
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0:
                gpu = {"available": True, "devices": r.stdout.strip().splitlines()}
        except (subprocess.TimeoutExpired, OSError):
            gpu = {"available": False, "error": "nvidia-smi failed"}

    return {
        "status": "ok" if sana_ok else "degraded",
        "sana": {"ready": sana_ok, "note": sana_note},
        "sana_repo": str(settings.sana_repo),
        "gpu": gpu,
    }
