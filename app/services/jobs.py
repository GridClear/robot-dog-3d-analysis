"""SQLite job registry and state transitions."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from app.config import get_settings
from app.schemas import JobArtifacts, JobError, JobRecord, JobStage, JobState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    state TEXT NOT NULL,
    stage TEXT,
    progress_pct REAL NOT NULL DEFAULT 0,
    n_frames INTEGER NOT NULL DEFAULT 0,
    pose_backend TEXT NOT NULL DEFAULT 'vggt',
    train_backend TEXT NOT NULL DEFAULT 'gsplat',
    max_iterations INTEGER,
    artifacts_json TEXT NOT NULL DEFAULT '{}',
    error_json TEXT,
    timings_json TEXT NOT NULL DEFAULT '{}'
);
"""


async def init_db() -> None:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(_SCHEMA)
        await db.commit()


def _parse_record(row: aiosqlite.Row) -> JobRecord:
    artifacts = JobArtifacts.model_validate(json.loads(row["artifacts_json"]))
    error = JobError.model_validate(json.loads(row["error_json"])) if row["error_json"] else None
    timings = json.loads(row["timings_json"])
    return JobRecord(
        job_id=row["job_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        state=JobState(row["state"]),
        stage=JobStage(row["stage"]) if row["stage"] else None,
        progress_pct=row["progress_pct"],
        n_frames=row["n_frames"],
        pose_backend=row["pose_backend"],
        train_backend=row["train_backend"],
        max_iterations=row["max_iterations"],
        artifacts=artifacts,
        error=error,
        timings_ms=timings,
    )


async def create_job(
    job_id: str,
    n_frames: int,
    *,
    pose_backend: str | None = None,
    train_backend: str | None = None,
    max_iterations: int | None = None,
) -> JobRecord:
    settings = get_settings()
    now = datetime.now(UTC).isoformat()
    record = JobRecord(
        job_id=job_id,
        created_at=datetime.fromisoformat(now),
        state=JobState.QUEUED,
        n_frames=n_frames,
        pose_backend=pose_backend or settings.pose_backend,
        train_backend=train_backend or settings.train_backend,
        max_iterations=max_iterations,
    )
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            """
            INSERT INTO jobs (job_id, created_at, state, n_frames, pose_backend, train_backend,
                              max_iterations, artifacts_json, timings_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.job_id,
                now,
                record.state.value,
                record.n_frames,
                record.pose_backend,
                record.train_backend,
                record.max_iterations,
                record.artifacts.model_dump_json(),
                json.dumps(record.timings_ms),
            ),
        )
        await db.commit()
    return record


async def get_job(job_id: str) -> JobRecord | None:
    settings = get_settings()
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = await cur.fetchone()
    return _parse_record(row) if row else None


async def update_job(record: JobRecord) -> None:
    settings = get_settings()
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            """
            UPDATE jobs SET state=?, stage=?, progress_pct=?, n_frames=?, pose_backend=?,
                train_backend=?, max_iterations=?, artifacts_json=?, error_json=?, timings_json=?
            WHERE job_id=?
            """,
            (
                record.state.value,
                record.stage.value if record.stage else None,
                record.progress_pct,
                record.n_frames,
                record.pose_backend,
                record.train_backend,
                record.max_iterations,
                record.artifacts.model_dump_json(),
                record.error.model_dump_json() if record.error else None,
                json.dumps(record.timings_ms),
                record.job_id,
            ),
        )
        await db.commit()


async def list_queued_jobs() -> list[str]:
    settings = get_settings()
    async with aiosqlite.connect(settings.db_path) as db:
        cur = await db.execute(
            "SELECT job_id FROM jobs WHERE state = ? ORDER BY created_at",
            (JobState.QUEUED.value,),
        )
        rows = await cur.fetchall()
    return [r[0] for r in rows]


async def reset_failed_to_queued(job_id: str) -> JobRecord | None:
    record = await get_job(job_id)
    if record is None or record.state != JobState.FAILED:
        return record
    record.state = JobState.QUEUED
    record.stage = None
    record.progress_pct = 0.0
    record.error = None
    await update_job(record)
    return record


def write_traceback(job_id: str, stage: str, exc: BaseException) -> Path:
    from app.services import storage

    log_dir = storage.logs_dir(job_id)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{stage}_error.txt"
    import traceback

    path.write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return path
