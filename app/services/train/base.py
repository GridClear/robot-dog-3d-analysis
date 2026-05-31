"""Gaussian splat training backend protocol."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class TrainResult:
    available: bool
    method: str
    ply_path: Path | None = None
    result_dir: Path | None = None
    log_path: Path | None = None
    note: str = ""


class TrainBackend(Protocol):
    name: str

    def is_available(self) -> tuple[bool, str]: ...

    def fit(
        self,
        colmap_dir: Path,
        out_dir: Path,
        *,
        max_iterations: int,
        data_factor: int,
        test_every: int,
        log_path: Path,
    ) -> TrainResult: ...
