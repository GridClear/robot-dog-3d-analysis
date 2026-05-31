"""World-model streaming sessions API."""
from __future__ import annotations

import asyncio
import json
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.config import get_settings
from app.motion_presets import PRESETS
from app.schemas import (
    ConfigResponse,
    CreateSessionResponse,
    SessionRecord,
    StartSessionResponse,
)
from app.services import sana_runner, storage, stream_worker

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


@router.get("/config")
async def get_config() -> ConfigResponse:
    s = get_settings()
    ready, note = sana_runner.sana_ready()
    return ConfigResponse(
        interval_sec=s.interval_sec,
        num_frames=s.capped_num_frames,
        fps=float(s.fps),
        max_clip_seconds=s.max_clip_seconds,
        use_refiner=s.use_refiner,
        nvfp4_enabled=s.nvfp4_enabled,
        motion_presets=sorted(PRESETS.keys()),
        sana_ready=ready,
        sana_note=note,
    )


@router.post("", response_model=CreateSessionResponse)
async def create_session(
    files: Annotated[list[UploadFile], File()],
    prompt: Annotated[str | None, Form()] = None,
    action: Annotated[str | None, Form()] = None,
    loop: Annotated[bool, Form()] = False,
) -> CreateSessionResponse:
    if not files:
        raise HTTPException(400, "at least one image required")
    settings = get_settings()
    if len(files) > settings.max_images:
        raise HTTPException(400, f"max {settings.max_images} images")

    images: list[tuple[str, bytes]] = []
    for f in files:
        data = await f.read()
        if not data:
            continue
        images.append((f.filename or "image.jpg", data))
    if not images:
        raise HTTPException(400, "no valid image data")

    session_id = storage.new_session_id()
    text = (prompt or "").strip() or settings.default_prompt
    act = (action or "").strip() or settings.action_preset
    storage.save_prompt(session_id, text)
    paths = storage.save_images(session_id, images)
    storage.init_session_meta(session_id, text, act, loop, paths)

    return CreateSessionResponse(
        session_id=session_id,
        n_images=len(paths),
        interval_sec=settings.interval_sec,
    )


@router.get("/{session_id}", response_model=SessionRecord)
async def get_session(session_id: str) -> SessionRecord:
    record = storage.read_meta(session_id)
    if not record:
        raise HTTPException(404, "session not found")
    return record


@router.post("/{session_id}/start", response_model=StartSessionResponse)
async def start_session(session_id: str) -> StartSessionResponse:
    record = storage.read_meta(session_id)
    if not record:
        raise HTTPException(404, "session not found")
    ready, note = sana_runner.sana_ready()
    if not ready:
        raise HTTPException(503, f"SANA-WM not ready: {note}")
    try:
        await stream_worker.start_session(session_id)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    record = storage.read_meta(session_id)
    assert record
    return StartSessionResponse(session_id=session_id, state=record.state)


@router.post("/{session_id}/stop")
async def stop_session(session_id: str) -> dict:
    record = storage.read_meta(session_id)
    if not record:
        raise HTTPException(404, "session not found")
    await stream_worker.stop_session(session_id)
    record = storage.read_meta(session_id)
    return {"session_id": session_id, "state": record.state.value if record else "unknown"}


@router.get("/{session_id}/events")
async def session_events(session_id: str) -> StreamingResponse:
    record = storage.read_meta(session_id)
    if not record:
        raise HTTPException(404, "session not found")

    q = stream_worker.subscribe(session_id)

    async def gen():
        try:
            yield f"data: {json.dumps({'type': 'hello', 'session_id': session_id, 'state': record.state.value})}\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            stream_worker.unsubscribe(session_id, q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/{session_id}/splat/latest.splat")
async def latest_splat(session_id: str) -> FileResponse:
    path = storage.latest_splat_path(session_id)
    if not path.is_file():
        raise HTTPException(404, "no splat yet")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename="latest.splat",
        headers={"Cache-Control": "no-store"},
    )
