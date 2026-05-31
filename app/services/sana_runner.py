"""Subprocess wrapper for SANA-WM inference."""
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
    output_mp4: Path | None
    elapsed_sec: float
    log_path: Path
    error: str | None = None


def sana_ready(settings: Settings | None = None) -> tuple[bool, str]:
    s = settings or get_settings()
    script = s.inference_script
    if not script.is_file():
        return False, f"Missing inference script: {script}"
    conda = shutil.which("conda")
    if not conda:
        return False, "conda not found on PATH"
    return True, "ok"


def _build_command(
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

    env_prefix = ["env", "DISABLE_XFORMERS=1"]
    return [
        conda,
        "run",
        "-n",
        settings.sana_conda_env,
        "--no-capture-output",
        *env_prefix,
        "python",
        *py_args,
    ]


def _find_output_mp4(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    mp4s = sorted(output_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return mp4s[0] if mp4s else None


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
    settings = get_settings()
    ready, note = sana_ready(settings)
    if not ready:
        return InferenceResult(False, None, 0.0, Path(), note)

    work_dir = dest_tmp.parent / f"work_{image_index:04d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = get_settings().data_dir / "sessions" / session_id / "logs" / f"inference_{image_index:04d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = _build_command(settings, image, prompt_file, action, work_dir)
    t0 = time.perf_counter()

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(settings.sana_repo),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    log_lines: list[str] = []

    async def _drain() -> None:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode(errors="replace").rstrip()
            log_lines.append(text)
            logger.info("[sana-wm %s] %s", session_id, text)

    drain_task = asyncio.create_task(_drain())
    try:
        while True:
            if cancel_event and cancel_event.is_set():
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                elapsed = time.perf_counter() - t0
                log_path.write_text("\n".join(log_lines) + "\n[cancelled]\n", encoding="utf-8")
                return InferenceResult(False, None, elapsed, log_path, "cancelled")

            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
                break
            except asyncio.TimeoutError:
                if time.perf_counter() - t0 > settings.inference_timeout_sec:
                    proc.kill()
                    await proc.wait()
                    elapsed = time.perf_counter() - t0
                    log_path.write_text("\n".join(log_lines) + "\n[timeout]\n", encoding="utf-8")
                    return InferenceResult(False, None, elapsed, log_path, "inference timeout")
    finally:
        await drain_task

    elapsed = time.perf_counter() - t0
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if proc.returncode != 0:
        return InferenceResult(
            False,
            None,
            elapsed,
            log_path,
            f"inference exited {proc.returncode}",
        )

    src = _find_output_mp4(work_dir)
    if not src:
        return InferenceResult(False, None, elapsed, log_path, "no mp4 in output_dir")

    dest_tmp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_tmp)
    return InferenceResult(True, dest_tmp, elapsed, log_path, None)


def promote_video(tmp_path: Path, final_path: Path, latest_path: Path) -> None:
    """Atomically promote temp mp4 to generation artifact and latest."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        final_path.unlink()
    tmp_path.replace(final_path)
    latest_path.write_bytes(final_path.read_bytes())
