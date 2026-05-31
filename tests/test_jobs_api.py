"""HTTP job flow tests (no GPU)."""
from __future__ import annotations

import io

import pytest
from PIL import Image


def _jpeg_bytes(color=(200, 50, 50)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), color).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_health(app_client):
    r = await app_client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_backends_status(app_client):
    r = await app_client.get("/v1/backends/status")
    assert r.status_code == 200
    body = r.json()
    assert "pose" in body and "train" in body


@pytest.mark.asyncio
async def test_ingest_run_status(app_client):
    files = [
        ("files", ("a.jpg", _jpeg_bytes((200, 0, 0)), "image/jpeg")),
        ("files", ("b.jpg", _jpeg_bytes((0, 200, 0)), "image/jpeg")),
        ("files", ("c.jpg", _jpeg_bytes((0, 0, 200)), "image/jpeg")),
    ]
    r = await app_client.post("/v1/jobs/ingest", files=files)
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert r.json()["n_frames"] == 3

    r = await app_client.post(f"/v1/jobs/{job_id}/run", json={})
    assert r.status_code == 200
    assert r.json()["state"] == "queued"

    import asyncio

    for _ in range(50):
        r = await app_client.get(f"/v1/jobs/{job_id}")
        assert r.status_code == 200
        if r.json()["state"] == "succeeded":
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail("job did not succeed")

    body = r.json()
    assert body["progress_pct"] == 100.0
    assert body["timings_ms"]["total_ms"] >= 0


@pytest.mark.asyncio
async def test_ingest_too_few_frames(app_client):
    files = [("files", ("a.jpg", _jpeg_bytes(), "image/jpeg"))]
    r = await app_client.post("/v1/jobs/ingest", files=files)
    assert r.status_code == 400
