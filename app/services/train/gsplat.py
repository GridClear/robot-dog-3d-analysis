"""gsplat simple_trainer subprocess wrapper."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from app.services.train.base import TrainResult


class GSplatTrainBackend:
    name = "gsplat"

    def is_available(self) -> tuple[bool, str]:
        try:
            import gsplat  # noqa: F401

            return True, "gsplat installed"
        except Exception as e:
            return False, f"gsplat not installed: {e}. Run scripts/install.sh"

    def fit(
        self,
        colmap_dir: Path,
        out_dir: Path,
        *,
        max_iterations: int,
        data_factor: int,
        test_every: int,
        log_path: Path,
    ) -> TrainResult:
        ready, note = self.is_available()
        if not ready:
            return TrainResult(False, self.name, note=note)

        trainer = self._find_trainer_script()
        if trainer is None:
            return TrainResult(
                False,
                self.name,
                note="simple_trainer.py not found; install gsplat from git",
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(trainer),
            "default",
            "--data_dir",
            str(colmap_dir),
            "--data_factor",
            str(data_factor),
            "--result_dir",
            str(out_dir),
            "--max_steps",
            str(max_iterations),
            "--test_every",
            str(test_every),
            "--disable_viewer",
        ]
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(trainer.parent),
            )

        if proc.returncode != 0:
            return TrainResult(
                False,
                self.name,
                result_dir=out_dir,
                log_path=log_path,
                note=f"trainer exited {proc.returncode}; see log",
            )

        ply = self._find_output_ply(out_dir)
        if ply is None:
            return TrainResult(
                False,
                self.name,
                result_dir=out_dir,
                log_path=log_path,
                note="training finished but no Gaussian PLY found",
            )
        return TrainResult(True, self.name, ply_path=ply, result_dir=out_dir, log_path=log_path)

    @staticmethod
    def _find_trainer_script() -> Path | None:
        project_root = Path(__file__).resolve().parents[3]
        candidates = [
            project_root / "vendor" / "gsplat" / "examples" / "simple_trainer.py",
            project_root / "vendor" / "simple_trainer.py",
        ]
        try:
            import gsplat

            pkg_root = Path(gsplat.__file__).resolve().parent.parent
            candidates.insert(0, pkg_root / "examples" / "simple_trainer.py")
        except Exception:
            pass
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _find_output_ply(out_dir: Path) -> Path | None:
        candidates = list(out_dir.rglob("*.ply"))
        if not candidates:
            return None
        gaussian = [p for p in candidates if "point_cloud" in str(p) or "splat" in p.name.lower()]
        pool = gaussian if gaussian else candidates
        return max(pool, key=lambda p: p.stat().st_mtime)
