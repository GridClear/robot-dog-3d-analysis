"""API and pipeline data contracts."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobStage(StrEnum):
    EXTRACT = "extract"
    POSE = "pose"
    TRAIN = "train"
    EXPORT = "export"
    DONE = "done"


class JobError(BaseModel):
    stage: str
    message: str
    traceback_path: str | None = None


class JobArtifacts(BaseModel):
    ply: str | None = None
    splat: str | None = None
    colmap_dir: str | None = None
    cameras_json: str | None = None
    preview_jpg: str | None = None
    log_path: str | None = None


class JobRecord(BaseModel):
    job_id: str
    created_at: datetime
    state: JobState
    stage: JobStage | None = None
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    n_frames: int = 0
    pose_backend: str = "vggt"
    train_backend: str = "gsplat"
    max_iterations: int | None = None
    artifacts: JobArtifacts = Field(default_factory=JobArtifacts)
    error: JobError | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    job_id: str
    n_frames: int


class RunJobRequest(BaseModel):
    pose_backend: str | None = None
    train_backend: str | None = None
    max_iterations: int | None = None


class RunJobResponse(BaseModel):
    job_id: str
    state: JobState


class BackendStatus(BaseModel):
    name: str
    ready: bool
    note: str = ""


class BackendsStatusResponse(BaseModel):
    pose: list[BackendStatus]
    train: list[BackendStatus]
    gpu: dict[str, Any] = Field(default_factory=dict)
