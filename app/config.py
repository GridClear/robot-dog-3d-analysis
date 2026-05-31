"""Runtime configuration for the SANA-WM streaming service."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WM_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8090
    data_dir: Path = Path("data")

    interval_sec: float = 30.0
    num_frames: int = 321
    fps: int = 16
    use_refiner: bool = False
    inference_step: int = 60
    nvfp4_enabled: bool = False
    offload_vae: bool = False
    offload_refiner: bool = False

    action_preset: str = "forward_explore"
    translation_speed: float = 0.055
    rotation_speed_deg: float = 1.2

    sana_conda_env: str = "sana"
    sana_repo: Path = Path("third_party/Sana")
    # Skip Pi3X intrinsics download when set (faster; less accurate for arbitrary photos)
    default_intrinsics: Path | None = None
    inference_timeout_sec: int = 3600

    # Splat lifting (Pi3X) — runs after each SANA-WM clip completes.
    splat_num_views: int = 8
    splat_max_gaussians: int = 250_000
    splat_conf_threshold: float = 0.30
    splat_timeout_sec: int = 600

    max_clip_seconds: float = 60.0
    max_image_height: int = 720
    max_images: int = 64

    session_max_age_hours: int = 72

    default_prompt: str = (
        "A first-person view from a robot dog camera exploring an indoor or outdoor "
        "environment. Natural lighting, realistic textures, stable scene geometry."
    )

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def benchmark_path(self) -> Path:
        return self.data_dir / "benchmark.json"

    @property
    def capped_num_frames(self) -> int:
        cap = int(self.max_clip_seconds * self.fps)
        return min(self.num_frames, cap)

    @property
    def inference_script(self) -> Path:
        return self.sana_repo / "inference_video_scripts" / "inference_sana_wm.py"

    @property
    def resolved_default_intrinsics(self) -> Path | None:
        if self.default_intrinsics and self.default_intrinsics.is_file():
            return self.default_intrinsics
        demo = self.sana_repo / "asset" / "sana_wm" / "demo_0_intrinsics.npy"
        return demo if demo.is_file() else None


def _apply_benchmark_overrides(s: Settings) -> Settings:
    import json

    path = s.benchmark_path
    if not path.is_file():
        return s
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rec = data.get("recommended") or {}
        if "interval_sec" in rec:
            s.interval_sec = float(rec["interval_sec"])
        if "num_frames" in rec:
            s.num_frames = int(rec["num_frames"])
        if "use_refiner" in rec:
            s.use_refiner = bool(rec["use_refiner"])
        if "inference_step" in rec:
            s.inference_step = int(rec["inference_step"])
        if rec.get("nvfp4_enabled") is True:
            s.nvfp4_enabled = True
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        pass
    return s


@lru_cache
def get_settings() -> Settings:
    s = _apply_benchmark_overrides(Settings())
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.sessions_dir.mkdir(parents=True, exist_ok=True)
    return s


def clear_settings_cache() -> None:
    get_settings.cache_clear()
