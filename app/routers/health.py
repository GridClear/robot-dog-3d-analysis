"""Liveness and backend status."""
from __future__ import annotations

from fastapi import APIRouter

from app.services.backends import backends_status

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/v1/backends/status")
async def status():
    return backends_status()
