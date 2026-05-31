"""API data contracts for world-model streaming sessions."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SessionState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class ImageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class ImageQueueItem(BaseModel):
    index: int
    filename: str
    status: ImageStatus = ImageStatus.PENDING
    error: str | None = None
    elapsed_sec: float | None = None
    n_gaussians: int | None = None


class SessionRecord(BaseModel):
    session_id: str
    created_at: datetime
    state: SessionState
    prompt: str
    action: str
    loop: bool = False
    n_images: int = 0
    current_index: int | None = None
    generation_id: int = 0
    images: list[ImageQueueItem] = Field(default_factory=list)
    error: str | None = None
    interval_sec: float = 30.0
    config: dict[str, Any] = Field(default_factory=dict)


class CreateSessionResponse(BaseModel):
    session_id: str
    n_images: int
    interval_sec: float


class StartSessionResponse(BaseModel):
    session_id: str
    state: SessionState


class ConfigResponse(BaseModel):
    interval_sec: float
    num_frames: int
    fps: float
    max_clip_seconds: float
    use_refiner: bool
    nvfp4_enabled: bool
    motion_presets: list[str]
    sana_ready: bool
    sana_note: str = ""
