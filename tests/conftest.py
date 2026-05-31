"""Pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import clear_settings_cache


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("WM_DATA_DIR", str(tmp_path / "data"))
    clear_settings_cache()
    yield tmp_path
    clear_settings_cache()
