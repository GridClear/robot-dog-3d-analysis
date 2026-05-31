"""Runtime configuration for the splat reconstruction service."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SPLAT_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8090
    data_dir: Path = Path("data")

    min_frames: int = 3
    max_frames: int = 120
    video_fps: int = 3

    pose_backend: str = "vggt"
    train_backend: str = "gsplat"

    train_iterations: int = 7000
    data_factor: int = 2
    test_every: int = 8

    job_max_age_hours: int = 72

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jobs.db"


@lru_cache
def get_settings() -> Settings:
    s = get_settings_uncached()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.sessions_dir.mkdir(parents=True, exist_ok=True)
    return s


def get_settings_uncached() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
