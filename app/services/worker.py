"""Background worker: single-GPU job queue."""
from __future__ import annotations

import asyncio
import logging

from app import pipeline
from app.config import get_settings
from app.schemas import JobError, JobState
from app.services import jobs
from app.services import storage

logger = logging.getLogger(__name__)

_queue: asyncio.Queue[str] | None = None
_gpu_lock: asyncio.Lock | None = None
_worker_task: asyncio.Task | None = None


def _queue_ref() -> asyncio.Queue[str]:
    global _queue
    if _queue is None:
        _queue = asyncio.Queue()
    return _queue


def _lock_ref() -> asyncio.Lock:
    global _gpu_lock
    if _gpu_lock is None:
        _gpu_lock = asyncio.Lock()
    return _gpu_lock


async def start_worker() -> None:
    global _worker_task, _queue, _gpu_lock
    _queue = asyncio.Queue()
    _gpu_lock = asyncio.Lock()
    await jobs.init_db()
    settings = get_settings()
    removed = storage.cleanup_old_sessions(settings.job_max_age_hours)
    if removed:
        logger.info("cleaned up %d old session dirs", removed)
    for job_id in await jobs.list_queued_jobs():
        await _queue_ref().put(job_id)
    _worker_task = asyncio.create_task(_worker_loop())


async def stop_worker() -> None:
    global _worker_task, _queue, _gpu_lock
    if _worker_task:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
        _worker_task = None
    _queue = None
    _gpu_lock = None


async def enqueue(job_id: str) -> None:
    await _queue_ref().put(job_id)


async def _worker_loop() -> None:
    q = _queue_ref()
    while True:
        job_id = await q.get()
        try:
            async with _lock_ref():
                record = await jobs.get_job(job_id)
                if record is None or record.state not in (JobState.QUEUED, JobState.FAILED):
                    continue
                if record.state == JobState.FAILED:
                    record = await jobs.reset_failed_to_queued(job_id)
                    if record is None:
                        continue
                await pipeline.run_job(record)
        except Exception:
            logger.exception("worker failed for job %s", job_id)
            record = await jobs.get_job(job_id)
            if record:
                tb = jobs.write_traceback(job_id, "worker", Exception("worker crash"))
                record.state = JobState.FAILED
                record.error = JobError(
                    stage="worker", message="unexpected worker error", traceback_path=str(tb)
                )
                await jobs.update_job(record)
        finally:
            q.task_done()
