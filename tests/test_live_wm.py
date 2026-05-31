"""Live SANA-WM smoke. Skipped unless WM_LIVE=1."""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("WM_LIVE") != "1",
    reason="set WM_LIVE=1 for GPU / conda tests",
)


@pytest.mark.live
def test_sana_ready():
    from app.services.sana_runner import sana_ready

    ok, note = sana_ready()
    assert ok, note
