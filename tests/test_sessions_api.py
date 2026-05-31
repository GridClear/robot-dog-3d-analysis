"""API tests with mocked SANA-WM inference."""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

@pytest.fixture
def client(tmp_path_factory, monkeypatch):
    data = tmp_path_factory.mktemp("wm_data")
    monkeypatch.setenv("WM_DATA_DIR", str(data))
    from app.config import clear_settings_cache

    clear_settings_cache()
    from app.main import app

    with TestClient(app) as c:
        yield c


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (100, 120, 140)).save(buf, "JPEG")
    return buf.getvalue()


def test_config(client: TestClient):
    with patch("app.services.sana_runner.sana_ready", return_value=(True, "ok")):
        r = client.get("/v1/sessions/config")
    assert r.status_code == 200
    body = r.json()
    assert "interval_sec" in body
    assert "forward_explore" in body["motion_presets"]


def test_create_session(client: TestClient):
    r = client.post(
        "/v1/sessions",
        files=[("files", ("a.jpg", _jpeg_bytes(), "image/jpeg"))],
        data={"prompt": "test scene", "action": "forward_explore"},
    )
    assert r.status_code == 200
    sid = r.json()["session_id"]
    meta = client.get(f"/v1/sessions/{sid}")
    assert meta.status_code == 200
    assert meta.json()["n_images"] == 1


async def _inline_start(session_id: str) -> None:
    import asyncio

    from app.services import stream_worker

    cancel = asyncio.Event()
    stream_worker._cancel_events[session_id] = cancel
    await stream_worker._run_loop(session_id, cancel)


def test_start_mocked_inference(client: TestClient, tmp_path: Path):
    from app.services import sana_runner

    async def fake_run(session_id, image_index, image, prompt_file, action, dest_tmp, **kwargs):
        dest_tmp.parent.mkdir(parents=True, exist_ok=True)
        dest_tmp.write_bytes(b"\x00\x00\x00\x00")
        return sana_runner.InferenceResult(True, dest_tmp, 0.5, tmp_path / "log.txt", None)

    with patch("app.services.sana_runner.sana_ready", return_value=(True, "ok")):
        with patch("app.services.sana_runner.run_inference", new=AsyncMock(side_effect=fake_run)):
            with patch("app.services.stream_worker.start_session", new=_inline_start):
                with patch("app.services.stream_worker.asyncio.sleep", new=AsyncMock()):
                    cr = client.post(
                        "/v1/sessions",
                        files=[("files", ("a.jpg", _jpeg_bytes(), "image/jpeg"))],
                        data={"prompt": "test"},
                    )
                    sid = cr.json()["session_id"]
                    sr = client.post(f"/v1/sessions/{sid}/start")
                    assert sr.status_code == 200
                    st = client.get(f"/v1/sessions/{sid}").json()
                    assert st["state"] == "completed"
