"""MASt3R-SfM pose backend (optional; aarch64 install may fail on GB10)."""
from __future__ import annotations

from pathlib import Path

from app.services.pose.base import PoseResult


class MASt3RPoseBackend:
    name = "mast3r"

    def is_available(self) -> tuple[bool, str]:
        try:
            import mast3r  # noqa: F401

            return True, "mast3r import ok"
        except Exception as e:
            return False, (
                f"mast3r not installed: {e}. "
                "Clone https://github.com/naver/mast3r and install per upstream docs."
            )

    def estimate(self, frames: list[Path], work_dir: Path) -> PoseResult:
        ready, note = self.is_available()
        if not ready:
            return PoseResult(False, self.name, len(frames), note=note)

        # Full MASt3R-SfM integration is environment-specific; delegate when installed.
        return PoseResult(
            False,
            self.name,
            len(frames),
            note="MASt3R adapter stub: install mast3r and wire SfM export in a future update",
        )
