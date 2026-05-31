"""Subprocess wrapper for SANA-WM inference + Pi3X-based splat lifting."""
from __future__ import annotations

import asyncio
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app.motion_presets import resolve_action

logger = logging.getLogger(__name__)

_nvfp4_support_cache: bool | None = None
REPO_ROOT = Path(__file__).resolve().parents[2]
SPLAT_BUILDER = REPO_ROOT / "scripts" / "build_splat.py"


def inference_supports_nvfp4(settings: Settings | None = None) -> bool:
    """True when upstream inference_sana_wm.py exposes --nvfp4."""
    global _nvfp4_support_cache
    if _nvfp4_support_cache is not None:
        return _nvfp4_support_cache
    s = settings or get_settings()
    script = s.inference_script
    if not script.is_file():
        _nvfp4_support_cache = False
        return False
    text = script.read_text(encoding="utf-8", errors="ignore")
    _nvfp4_support_cache = '"--nvfp4"' in text or "'--nvfp4'" in text
    return _nvfp4_support_cache


@dataclass
class InferenceResult:
    ok: bool
    output_splat: Path | None
    elapsed_sec: float
    log_path: Path
    error: str | None = None
    n_gaussians: int = 0


def sana_ready(settings: Settings | None = None) -> tuple[bool, str]:
    s = settings or get_settings()
    script = s.inference_script
    if not script.is_file():
        return False, f"Missing inference script: {script}"
    if not SPLAT_BUILDER.is_file():
        return False, f"Missing splat builder: {SPLAT_BUILDER}"
    conda = shutil.which("conda")
    if not conda:
        return False, "conda not found on PATH"
    return True, "ok"


def _build_sana_cmd(
    settings: Settings,
    image: Path,
    prompt_file: Path,
    action: str,
    output_dir: Path,
) -> list[str]:
    conda = shutil.which("conda")
    if not conda:
        raise RuntimeError("conda not found on PATH")

    py_args = [
        str(settings.inference_script),
        "--image",
        str(image),
        "--prompt",
        str(prompt_file),
        "--action",
        resolve_action(action),
        "--translation_speed",
        str(settings.translation_speed),
        "--rotation_speed_deg",
        str(settings.rotation_speed_deg),
        "--num_frames",
        str(settings.capped_num_frames),
        "--fps",
        str(settings.fps),
        "--step",
        str(settings.inference_step),
        "--output_dir",
        str(output_dir),
        "--no_action_overlay",
    ]
    if not settings.use_refiner:
        py_args.append("--no_refiner")
    if settings.offload_vae:
        py_args.append("--offload_vae")
    if settings.offload_refiner:
        py_args.append("--offload_refiner")
    intr = settings.resolved_default_intrinsics
    if intr is not None:
        py_args.extend(["--intrinsics", str(intr)])
    if settings.nvfp4_enabled:
        if inference_supports_nvfp4(settings):
            py_args.append("--nvfp4")
        else:
            logger.warning("WM_NVFP4_ENABLED=1 but inference script has no --nvfp4 yet")

    return [
        conda,
        "run",
        "-n",
        settings.sana_conda_env,
        "--no-capture-output",
        "env",
        "DISABLE_XFORMERS=1",
        "python",
        *py_args,
    ]


def _build_splat_cmd(
    settings: Settings,
    mp4: Path,
    seed_image: Path,
    splat_out: Path,
) -> list[str]:
    conda = shutil.which("conda")
    if not conda:
        raise RuntimeError("conda not found on PATH")
    return [
        conda,
        "run",
        "-n",
        settings.sana_conda_env,
        "--no-capture-output",
        "python",
        str(SPLAT_BUILDER),
        "--mp4",
        str(mp4),
        "--seed_image",
        str(seed_image),
        "--output",
        str(splat_out),
        "--num_views",
        str(settings.splat_num_views),
        "--max_gaussians",
        str(settings.splat_max_gaussians),
        "--conf_threshold",
        str(settings.splat_conf_threshold),
    ]


def _find_output_mp4(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    mp4s = sorted(output_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return mp4s[0] if mp4s else None


async def _run_subprocess(
    cmd: list[str],
    *,
    cwd: Path | None,
    log_lines: list[str],
    label: str,
    session_id: str,
    timeout_sec: int,
    cancel_event: asyncio.Event | None,
) -> int | None:
    """Run ``cmd`` streaming stdout to ``log_lines``; respect cancel/timeout."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    async def _drain() -> None:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode(errors="replace").rstrip()
            log_lines.append(f"[{label}] {text}")
            logger.info("[%s %s] %s", label, session_id, text)

    drain_task = asyncio.create_task(_drain())
    t0 = time.perf_counter()
    try:
        while True:
            if cancel_event and cancel_event.is_set():
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                return None
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
                break
            except asyncio.TimeoutError:
                if time.perf_counter() - t0 > timeout_sec:
                    proc.kill()
                    await proc.wait()
                    return None
    finally:
        await drain_task
    return proc.returncode


async def run_inference(
    session_id: str,
    image_index: int,
    image: Path,
    prompt_file: Path,
    action: str,
    dest_tmp: Path,
    *,
    cancel_event: asyncio.Event | None = None,
) -> InferenceResult:
    """Generate a SANA-WM video and lift it to a Gaussian splat.

    ``dest_tmp`` is the temp .splat path; the intermediate mp4 is kept
    alongside the artifact for debugging but is not promoted to the
    public ``latest`` slot.
    """
    settings = get_settings()
    ready, note = sana_ready(settings)
    if not ready:
        return InferenceResult(False, None, 0.0, Path(), note)

    work_dir = dest_tmp.parent / f"work_{image_index:04d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.data_dir / "sessions" / session_id / "logs" / f"inference_{image_index:04d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    t0 = time.perf_counter()

    sana_cmd = _build_sana_cmd(settings, image, prompt_file, action, work_dir)
    rc = await _run_subprocess(
        sana_cmd,
        cwd=settings.sana_repo,
        log_lines=log_lines,
        label="sana-wm",
        session_id=session_id,
        timeout_sec=settings.inference_timeout_sec,
        cancel_event=cancel_event,
    )
    if rc is None:
        elapsed = time.perf_counter() - t0
        log_path.write_text("\n".join(log_lines) + "\n[cancelled-or-timeout]\n", encoding="utf-8")
        return InferenceResult(False, None, elapsed, log_path, "cancelled or sana timeout")
    if rc != 0:
        elapsed = time.perf_counter() - t0
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return InferenceResult(False, None, elapsed, log_path, f"sana-wm exited {rc}")

    mp4 = _find_output_mp4(work_dir)
    if not mp4:
        elapsed = time.perf_counter() - t0
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return InferenceResult(False, None, elapsed, log_path, "no mp4 in output_dir")

    splat_cmd = _build_splat_cmd(settings, mp4, image, dest_tmp)
    rc = await _run_subprocess(
        splat_cmd,
        cwd=REPO_ROOT,
        log_lines=log_lines,
        label="splat",
        session_id=session_id,
        timeout_sec=settings.splat_timeout_sec,
        cancel_event=cancel_event,
    )
    elapsed = time.perf_counter() - t0
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if rc is None:
        return InferenceResult(False, None, elapsed, log_path, "cancelled or splat timeout")
    if rc != 0:
        return InferenceResult(False, None, elapsed, log_path, f"splat builder exited {rc}")
    if not dest_tmp.is_file() or dest_tmp.stat().st_size == 0:
        return InferenceResult(False, None, elapsed, log_path, "splat output missing or empty")

    n_gaussians = dest_tmp.stat().st_size // 32
    # Keep mp4 next to the splat for debugging; rename so generations don't collide.
    mp4_keep = dest_tmp.with_suffix(".mp4")
    try:
        shutil.copy2(mp4, mp4_keep)
    except OSError:
        pass

    return InferenceResult(True, dest_tmp, elapsed, log_path, None, n_gaussians)


def promote_splat(tmp_path: Path, final_path: Path, latest_path: Path) -> None:
    """Atomically promote temp .splat to generation artifact and latest."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        final_path.unlink()
    tmp_path.replace(final_path)
    latest_path.write_bytes(final_path.read_bytes())
