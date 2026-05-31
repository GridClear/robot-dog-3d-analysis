"""Pytest fixtures."""
from __future__ import annotations

import pytest

from app.config import clear_settings_cache


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_DATA_DIR", str(tmp_path / "data"))
    clear_settings_cache()
    yield tmp_path
    clear_settings_cache()


@pytest.fixture
async def app_client(tmp_data, monkeypatch):
    from httpx import ASGITransport, AsyncClient

    from app.main import app
    from app.schemas import JobRecord, JobStage, JobState
    from app.services import jobs, worker

    async def _stub_run(record: JobRecord) -> JobRecord:
        record.state = JobState.SUCCEEDED
        record.stage = JobStage.DONE
        record.progress_pct = 100.0
        record.timings_ms = {"pose_ms": 1, "train_ms": 2, "export_ms": 3, "total_ms": 6}
        await jobs.update_job(record)
        return record

    monkeypatch.setattr("app.pipeline.run_job", _stub_run)
    await worker.start_worker()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await worker.stop_worker()
