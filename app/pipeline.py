"""End-to-end job orchestration."""
from __future__ import annotations

import time
from pathlib import Path

from app.config import get_settings
from app.schemas import JobError, JobRecord, JobStage, JobState
from app.services import jobs
from app.services.backends import get_pose_backend, get_train_backend
from app.services.export import ply as ply_export
from app.services.export import splat as splat_export
from app.services import storage


async def run_job(record: JobRecord) -> JobRecord:
    settings = get_settings()
    timings: dict[str, float] = dict(record.timings_ms)
    t_total = time.perf_counter()

    frames = storage.frame_paths(record.job_id)
    if len(frames) < settings.min_frames:
        record.state = JobState.FAILED
        record.error = JobError(
            stage=JobStage.EXTRACT.value,
            message=f"need >= {settings.min_frames} frames, got {len(frames)}",
        )
        await jobs.update_job(record)
        return record

    record.n_frames = len(frames)
    record.state = JobState.RUNNING
    record.stage = JobStage.POSE
    record.progress_pct = 10.0
    await jobs.update_job(record)

    pose_backend = get_pose_backend(record.pose_backend)
    t0 = time.perf_counter()
    pose_result = pose_backend.estimate(frames, storage.session_dir(record.job_id))
    timings["pose_ms"] = round((time.perf_counter() - t0) * 1000)

    if not pose_result.available or pose_result.colmap_dir is None:
        record.state = JobState.FAILED
        record.error = JobError(stage=JobStage.POSE.value, message=pose_result.note)
        record.timings_ms = timings
        await jobs.update_job(record)
        return record

    record.artifacts.colmap_dir = str(pose_result.colmap_dir)
    if pose_result.cameras_json:
        dst_cam = storage.artifacts_dir(record.job_id) / "cameras.json"
        dst_cam.parent.mkdir(parents=True, exist_ok=True)
        dst_cam.write_bytes(pose_result.cameras_json.read_bytes())
        record.artifacts.cameras_json = str(dst_cam)

    record.stage = JobStage.TRAIN
    record.progress_pct = 40.0
    record.timings_ms = timings
    await jobs.update_job(record)

    train_backend = get_train_backend(record.train_backend)
    train_out = storage.train_dir(record.job_id)
    log_path = storage.logs_dir(record.job_id) / "train.log"
    max_iters = record.max_iterations or settings.train_iterations

    t0 = time.perf_counter()
    train_result = train_backend.fit(
        pose_result.colmap_dir,
        train_out,
        max_iterations=max_iters,
        data_factor=settings.data_factor,
        test_every=settings.test_every,
        log_path=log_path,
    )
    timings["train_ms"] = round((time.perf_counter() - t0) * 1000)

    if not train_result.available or train_result.ply_path is None:
        record.state = JobState.FAILED
        record.error = JobError(
            stage=JobStage.TRAIN.value,
            message=train_result.note,
            traceback_path=str(log_path) if log_path.exists() else None,
        )
        record.artifacts.log_path = str(log_path)
        record.timings_ms = timings
        await jobs.update_job(record)
        return record

    record.stage = JobStage.EXPORT
    record.progress_pct = 85.0
    record.timings_ms = timings
    await jobs.update_job(record)

    artifacts = storage.artifacts_dir(record.job_id)
    t0 = time.perf_counter()
    ply_dst = artifacts / "scene.ply"
    splat_dst = artifacts / "scene.splat"
    ply_export.export_ply(train_result.ply_path, ply_dst)
    try:
        splat_export.export_splat(ply_dst, splat_dst)
    except Exception as e:
        record.state = JobState.FAILED
        record.error = JobError(stage=JobStage.EXPORT.value, message=str(e))
        record.timings_ms = timings
        await jobs.update_job(record)
        return record

    timings["export_ms"] = round((time.perf_counter() - t0) * 1000)
    timings["total_ms"] = round((time.perf_counter() - t_total) * 1000)

    record.artifacts.ply = str(ply_dst)
    record.artifacts.splat = str(splat_dst)
    record.artifacts.log_path = str(log_path)
    record.stage = JobStage.DONE
    record.progress_pct = 100.0
    record.state = JobState.SUCCEEDED
    record.timings_ms = timings
    await jobs.update_job(record)
    return record
