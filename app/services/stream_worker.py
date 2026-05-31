"""Sequential SANA-WM generation queue with SSE event broadcast."""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import Any

from app.config import get_settings
from app.schemas import ImageStatus, SessionState
from app.services import sana_runner, storage

logger = logging.getLogger(__name__)

_gpu_lock = asyncio.Lock()
_running: dict[str, asyncio.Task] = {}
_cancel_events: dict[str, asyncio.Event] = {}
_subscribers: dict[str, list[asyncio.Queue[str]]] = defaultdict(list)


def _publish(session_id: str, event: dict[str, Any]) -> None:
    payload = json.dumps(event)
    for q in list(_subscribers.get(session_id, [])):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


def subscribe(session_id: str) -> asyncio.Queue[str]:
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=256)
    _subscribers[session_id].append(q)
    return q


def unsubscribe(session_id: str, q: asyncio.Queue[str]) -> None:
    subs = _subscribers.get(session_id, [])
    if q in subs:
        subs.remove(q)


async def start_session(session_id: str) -> None:
    if session_id in _running and not _running[session_id].done():
        return
    record = storage.read_meta(session_id)
    if not record:
        raise KeyError(session_id)
    if record.n_images == 0:
        raise ValueError("no images in session")

    cancel = asyncio.Event()
    _cancel_events[session_id] = cancel
    _running[session_id] = asyncio.create_task(_run_loop(session_id, cancel))


async def stop_session(session_id: str) -> None:
    ev = _cancel_events.get(session_id)
    if ev:
        ev.set()
    task = _running.get(session_id)
    if task:
        try:
            await asyncio.wait_for(task, timeout=15)
        except asyncio.TimeoutError:
            task.cancel()
    record = storage.read_meta(session_id)
    if record and record.state == SessionState.RUNNING:
        record.state = SessionState.STOPPED
        storage.write_meta(record)
    _publish(session_id, {"type": "stopped", "session_id": session_id})


async def _run_loop(session_id: str, cancel: asyncio.Event) -> None:
    settings = get_settings()
    record = storage.read_meta(session_id)
    if not record:
        return

    record.state = SessionState.RUNNING
    storage.write_meta(record)
    _publish(session_id, {"type": "started", "session_id": session_id, "interval_sec": record.interval_sec})

    images = storage.input_image_paths(session_id)
    prompt_file = storage.prompt_path(session_id)
    pass_num = 0

    try:
        while True:
            if cancel.is_set():
                break
            for index, image_path in enumerate(images):
                if cancel.is_set():
                    break

                record = storage.read_meta(session_id)
                if not record:
                    return

                record.current_index = index
                if index < len(record.images):
                    record.images[index].status = ImageStatus.RUNNING
                storage.write_meta(record)

                _publish(
                    session_id,
                    {
                        "type": "generation_started",
                        "session_id": session_id,
                        "index": index,
                        "pass": pass_num,
                    },
                )

                tmp = storage.generation_splat_temp_path(session_id, index)
                final = storage.generation_splat_path(session_id, index)
                latest = storage.latest_splat_path(session_id)

                async with _gpu_lock:
                    result = await sana_runner.run_inference(
                        session_id,
                        index,
                        image_path,
                        prompt_file,
                        record.action,
                        tmp,
                        cancel_event=cancel,
                    )

                record = storage.read_meta(session_id)
                if not record:
                    return

                if cancel.is_set() and not result.ok:
                    record.state = SessionState.STOPPED
                    storage.write_meta(record)
                    break

                if result.ok and result.output_splat:
                    sana_runner.promote_splat(result.output_splat, final, latest)
                    record.generation_id += 1
                    if index < len(record.images):
                        record.images[index].status = ImageStatus.DONE
                        record.images[index].elapsed_sec = result.elapsed_sec
                        record.images[index].n_gaussians = result.n_gaussians
                    storage.write_meta(record)

                    wait_sec = max(0.0, record.interval_sec - result.elapsed_sec)
                    _publish(
                        session_id,
                        {
                            "type": "generation_ready",
                            "session_id": session_id,
                            "index": index,
                            "generation_id": record.generation_id,
                            "url": f"/v1/sessions/{session_id}/splat/latest.splat",
                            "n_gaussians": result.n_gaussians,
                            "elapsed_sec": result.elapsed_sec,
                            "next_in_sec": wait_sec,
                        },
                    )
                    if wait_sec > 0 and not cancel.is_set():
                        await asyncio.sleep(wait_sec)
                else:
                    if index < len(record.images):
                        record.images[index].status = ImageStatus.FAILED
                        record.images[index].error = result.error or "unknown error"
                    record.state = SessionState.FAILED
                    record.error = result.error
                    storage.write_meta(record)
                    _publish(
                        session_id,
                        {
                            "type": "generation_failed",
                            "session_id": session_id,
                            "index": index,
                            "error": result.error,
                        },
                    )
                    return

            if cancel.is_set():
                break
            record = storage.read_meta(session_id)
            if not record or not record.loop:
                break
            pass_num += 1
            for item in record.images:
                item.status = ImageStatus.PENDING
                item.error = None
            storage.write_meta(record)

        record = storage.read_meta(session_id)
        if record:
            if cancel.is_set():
                record.state = SessionState.STOPPED
            elif record.state == SessionState.RUNNING:
                record.state = SessionState.COMPLETED
            storage.write_meta(record)
            _publish(
                session_id,
                {
                    "type": "completed" if record.state == SessionState.COMPLETED else "stopped",
                    "session_id": session_id,
                    "state": record.state.value,
                },
            )
    except asyncio.CancelledError:
        record = storage.read_meta(session_id)
        if record:
            record.state = SessionState.STOPPED
            storage.write_meta(record)
        raise
    finally:
        _running.pop(session_id, None)
        _cancel_events.pop(session_id, None)
