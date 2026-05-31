"""Gaussian PLY → antimatter15 .splat binary format."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np


def ply_to_splat(ply_path: Path) -> bytes:
    try:
        from plyfile import PlyData
    except ImportError as e:
        raise RuntimeError("plyfile required for splat export: pip install plyfile") from e

    plydata = PlyData.read(str(ply_path))
    vert = plydata["vertex"]
    sorted_indices = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
        / (1 + np.exp(-vert["opacity"]))
    )
    buffer = BytesIO()
    sh_c0 = 0.28209479177387814
    for idx in sorted_indices:
        v = plydata["vertex"][idx]
        position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
        scales = np.exp(
            np.array([v["scale_0"], v["scale_1"], v["scale_2"]], dtype=np.float32)
        )
        rot = np.array([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], dtype=np.float32)
        color = np.array(
            [
                0.5 + sh_c0 * v["f_dc_0"],
                0.5 + sh_c0 * v["f_dc_1"],
                0.5 + sh_c0 * v["f_dc_2"],
                1 / (1 + np.exp(-v["opacity"])),
            ]
        )
        buffer.write(position.tobytes())
        buffer.write(scales.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        norm = np.linalg.norm(rot) or 1.0
        buffer.write(((rot / norm) * 128 + 128).clip(0, 255).astype(np.uint8).tobytes())
    return buffer.getvalue()


def export_splat(ply_path: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(ply_to_splat(ply_path))
    return dst
