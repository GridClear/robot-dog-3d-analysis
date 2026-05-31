"""Job ingest, run, status, and artifact download."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import get_settings
from app.schemas import IngestResponse, JobRecord, JobState, RunJobRequest, RunJobResponse
from app.services import jobs, storage, worker
from app.services.video import dedupe_frames, extract_frames

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


async def _read_uploads(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    return [(f.filename or "frame.jpg", await f.read()) for f in files]


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    files: list[UploadFile] | None = File(None),
    video: UploadFile | None = File(None),
) -> IngestResponse:
    settings = get_settings()
    if not files and not video:
        raise HTTPException(400, "upload files[] and/or video")

    job_id = storage.new_job_id()
    frames: list[Path] = []

    if video is not None:
        vname = video.filename or "video.mp4"
        suffix = Path(vname).suffix.lower()
        if suffix not in _VIDEO_SUFFIXES:
            raise HTTPException(400, f"unsupported video type: {suffix}")
        vpath = storage.save_video(job_id, vname, await video.read())
        extracted = extract_frames(vpath, storage.frames_dir(job_id))
        frames = dedupe_frames(extracted, max_keep=settings.max_frames)

    if files:
        uploaded = await _read_uploads(files)
        saved = storage.save_frames(job_id, uploaded)
        if not frames:
            frames = saved
        else:
            offset = len(frames)
            for i, p in enumerate(saved):
                target = storage.frames_dir(job_id) / f"{offset + i:04d}{p.suffix}"
                target.write_bytes(p.read_bytes())
            frames = storage.frame_paths(job_id)

    if not frames:
        frames = storage.frame_paths(job_id)

    n = len(frames)
    if n < settings.min_frames:
        raise HTTPException(400, f"need >= {settings.min_frames} frames, got {n}")
    if n > settings.max_frames:
        raise HTTPException(400, f"max {settings.max_frames} frames, got {n}")

    await jobs.create_job(job_id, n)
    return IngestResponse(job_id=job_id, n_frames=n)


@router.post("/{job_id}/run", response_model=RunJobResponse)
async def run_job(job_id: str, body: RunJobRequest | None = None) -> RunJobResponse:
    record = await jobs.get_job(job_id)
    if record is None:
        raise HTTPException(404, "job not found")
    if record.state == JobState.RUNNING:
        raise HTTPException(409, "job already running")
    if record.state == JobState.SUCCEEDED:
        raise HTTPException(409, "job already succeeded")

    if body:
        if body.pose_backend:
            record.pose_backend = body.pose_backend
        if body.train_backend:
            record.train_backend = body.train_backend
        if body.max_iterations is not None:
            record.max_iterations = body.max_iterations

    record.state = JobState.QUEUED
    record.error = None
    await jobs.update_job(record)
    await worker.enqueue(job_id)
    return RunJobResponse(job_id=job_id, state=JobState.QUEUED)


@router.post("/{job_id}/retry", response_model=RunJobResponse)
async def retry_job(job_id: str) -> RunJobResponse:
    record = await jobs.reset_failed_to_queued(job_id)
    if record is None:
        raise HTTPException(404, "job not found or not failed")
    await worker.enqueue(job_id)
    return RunJobResponse(job_id=job_id, state=JobState.QUEUED)


@router.get("/{job_id}", response_model=JobRecord)
async def job_status(job_id: str) -> JobRecord:
    record = await jobs.get_job(job_id)
    if record is None:
        raise HTTPException(404, "job not found")
    return record


@router.get("/{job_id}/artifacts/{name}")
async def download_artifact(job_id: str, name: str) -> FileResponse:
    allowed = {"scene.ply", "scene.splat", "cameras.json", "preview.jpg"}
    if name not in allowed:
        raise HTTPException(404, "unknown artifact")
    path = storage.artifact_path(job_id, name)
    if path is None:
        raise HTTPException(404, "artifact not found")
    media = {
        "scene.ply": "application/octet-stream",
        "scene.splat": "application/octet-stream",
        "cameras.json": "application/json",
        "preview.jpg": "image/jpeg",
    }
    return FileResponse(path, media_type=media[name], filename=name)
