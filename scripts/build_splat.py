#!/usr/bin/env python
"""Lift a SANA-WM mp4 to a Niantic ``.splat`` Gaussian-splat file.

Runs inside the ``sana`` conda env (Pi3X + torch are required). The
pipeline samples K evenly-spaced frames from the world-model video,
feeds them to Pi3X as a single multi-view batch, takes the fused
world-space ``points`` output (already permutation-equivariant and
camera-consistent), filters by confidence, optionally prepends the
original seed image as an anchor frame, and writes the 32-byte/gaussian
binary container that antimatter15/splat, mkkellogg/GaussianSplats3D
and gsplat.js all consume directly.

The output is opaque and view-independent (no spherical harmonics) and
is intentionally minimal so the browser viewer can stream it on first
generation. Scene scale is set proportional to the bounding-box extent
so the splats render as small isotropic blobs rather than huge sheets.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_XFORMERS", "1")

import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms as T  # noqa: E402


PI3X_PIXEL_LIMIT = 255_000
IDENTITY_QUAT_BYTES = np.array([255, 128, 128, 128], dtype=np.uint8)


def _pi3x_size(w: int, h: int) -> tuple[int, int]:
    """Round (w, h) to the largest 14-multiple pair within Pi3X's pixel budget."""
    scale = math.sqrt(PI3X_PIXEL_LIMIT / max(w * h, 1))
    wt, ht = w * scale, h * scale
    k, m = max(1, round(wt / 14)), max(1, round(ht / 14))
    while (k * 14) * (m * 14) > PI3X_PIXEL_LIMIT:
        if k / m > wt / ht:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def _sample_frames(mp4: Path, num_frames: int) -> list[Image.Image]:
    raw = list(iio.imiter(str(mp4)))
    if not raw:
        raise SystemExit(f"[splat] no frames in {mp4}")
    total = len(raw)
    idx = (
        list(range(total))
        if total <= num_frames
        else np.linspace(0, total - 1, num_frames).round().astype(int).tolist()
    )
    return [Image.fromarray(raw[i]).convert("RGB") for i in idx]


@torch.inference_mode()
def _pi3x_reconstruct(
    images: list[Image.Image], device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(points[N,H,W,3], conf[N,H,W], rgb[N,H,W,3 uint8])`` in Pi3X world."""
    from pi3.models.pi3x import Pi3X

    w0, h0 = images[0].size
    w, h = _pi3x_size(w0, h0)
    resized = [im.resize((w, h), Image.Resampling.LANCZOS) for im in images]
    tensor = torch.stack([T.ToTensor()(im) for im in resized]).unsqueeze(0).to(device)

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
    model.disable_multimodal()
    model.requires_grad_(False)
    with torch.amp.autocast("cuda", dtype=dtype):
        out = model(imgs=tensor)

    pts = out["points"][0].float().cpu().numpy()
    conf = torch.sigmoid(out["conf"][0]).float().cpu().numpy()[..., 0]
    rgb = np.stack([np.asarray(im, dtype=np.uint8) for im in resized], axis=0)
    return pts, conf, rgb


def _opencv_to_opengl(pts: np.ndarray) -> np.ndarray:
    """Flip Y and Z so OpenCV (Y-down, Z-forward) becomes WebGL (Y-up, Z-back)."""
    out = pts.copy()
    out[..., 1] *= -1.0
    out[..., 2] *= -1.0
    return out


def _scene_scale(positions: np.ndarray) -> float:
    if positions.shape[0] < 2:
        return 0.01
    span = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    return max(span * 0.001, 1e-3)


def _pack_splat(positions: np.ndarray, colors: np.ndarray, scale: float) -> bytes:
    n = positions.shape[0]
    rec = np.zeros(
        n,
        dtype=np.dtype(
            [
                ("pos", "<f4", (3,)),
                ("scl", "<f4", (3,)),
                ("rgba", "u1", (4,)),
                ("quat", "u1", (4,)),
            ]
        ),
    )
    assert rec.dtype.itemsize == 32, rec.dtype.itemsize
    rec["pos"] = positions.astype(np.float32, copy=False)
    rec["scl"] = np.full((n, 3), scale, dtype=np.float32)
    rgba = np.empty((n, 4), dtype=np.uint8)
    rgba[:, :3] = colors.astype(np.uint8, copy=False)
    rgba[:, 3] = 255
    rec["rgba"] = rgba
    rec["quat"] = IDENTITY_QUAT_BYTES
    return rec.tobytes()


def _build(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = _sample_frames(args.mp4, args.num_views)
    if args.seed_image and args.seed_image.is_file():
        frames = [Image.open(args.seed_image).convert("RGB")] + frames

    pts, conf, rgb = _pi3x_reconstruct(frames, device)

    mask = conf >= args.conf_threshold
    positions = pts[mask]
    colors = rgb[mask]
    if positions.shape[0] == 0:
        flat = conf.reshape(-1)
        k = max(1, int(0.10 * flat.size))
        top = np.argpartition(flat, -k)[-k:]
        positions = pts.reshape(-1, 3)[top]
        colors = rgb.reshape(-1, 3)[top]

    if positions.shape[0] > args.max_gaussians:
        rng = np.random.default_rng(0)
        sel = rng.choice(positions.shape[0], size=args.max_gaussians, replace=False)
        positions, colors = positions[sel], colors[sel]

    positions = _opencv_to_opengl(positions)
    scale = _scene_scale(positions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = _pack_splat(positions, colors, scale)
    args.output.write_bytes(payload)

    meta_path = args.output.with_suffix(".splat.json")
    import json

    meta_path.write_text(
        json.dumps(
            {
                "n_gaussians": int(positions.shape[0]),
                "scale": scale,
                "source_mp4": str(args.mp4),
                "num_views": int(args.num_views),
                "conf_threshold": float(args.conf_threshold),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[splat] {positions.shape[0]:,} gaussians "
        f"({len(payload):,} bytes) scale={scale:.4f} -> {args.output}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_splat", description=__doc__)
    ap.add_argument("--mp4", type=Path, required=True, help="SANA-WM output video.")
    ap.add_argument(
        "--seed_image",
        type=Path,
        default=None,
        help="Original input image; prepended as Pi3X anchor frame.",
    )
    ap.add_argument("--output", type=Path, required=True, help="Destination .splat path.")
    ap.add_argument("--num_views", type=int, default=8)
    ap.add_argument("--max_gaussians", type=int, default=250_000)
    ap.add_argument("--conf_threshold", type=float, default=0.30)
    args = ap.parse_args()

    try:
        _build(args)
    except Exception as exc:
        print(f"[splat] FAILED: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
