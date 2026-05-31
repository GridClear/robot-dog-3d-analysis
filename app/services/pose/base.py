"""Pose estimation backend protocol."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class PoseResult:
    available: bool
    method: str
    n_views: int
    colmap_dir: Path | None = None
    cameras_json: Path | None = None
    init_points_ply: Path | None = None
    note: str = ""


class PoseBackend(Protocol):
    name: str

    def is_available(self) -> tuple[bool, str]: ...

    def estimate(self, frames: list[Path], work_dir: Path) -> PoseResult: ...
