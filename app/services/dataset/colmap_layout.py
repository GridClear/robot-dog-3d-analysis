"""Build COLMAP sparse reconstruction + images/ tree for gsplat."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np


def _write_ply(path: Path, pts: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, (x, y, z) in enumerate(pts):
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.4f} {y:.4f} {z:.4f}\n")


def write_colmap_from_vggt(
    frames: list[Path],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    points: np.ndarray,
    colmap_root: Path,
) -> tuple[Path, Path]:
    """Write images/, sparse/0/, init PLY, and cameras.json from VGGT outputs."""
    images_dir = colmap_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = colmap_root / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pycolmap
    except ImportError:
        return _write_colmap_text_fallback(
            frames, extrinsics, intrinsics, points, colmap_root, images_dir, sparse_dir
        )

    recon = pycolmap.Reconstruction()
    cam_ids: list[int] = []
    width = height = 0

    for i, frame in enumerate(frames):
        dst = images_dir / frame.name
        if not dst.exists():
            shutil.copy2(frame, dst)
        K = intrinsics[i]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        from PIL import Image

        with Image.open(frame) as im:
            width, height = im.size
        params = np.array([fx, fy, cx, cy], dtype=np.float64)
        cam_id = i + 1
        cam = pycolmap.Camera(
            model="PINHOLE",
            width=width,
            height=height,
            params=params,
            camera_id=cam_id,
        )
        recon.add_camera(cam)
        cam_ids.append(cam_id)

        ext = extrinsics[i]
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(ext[:3, :3]), ext[:3, 3]
        )
        img_id = i + 1
        image = pycolmap.Image(
            id=img_id,
            name=frame.name,
            camera_id=cam_id,
            cam_from_world=cam_from_world,
        )
        recon.add_image(image)

    if len(points) > 0:
        step = max(1, len(points) // 50000)
        sampled = points[::step]
        for pid, pt in enumerate(sampled, start=1):
            recon.add_point3D(pt.astype(np.float64), pycolmap.Track(), np.zeros(3, dtype=np.uint8))

    recon.write(sparse_dir)

    init_ply = colmap_root / "init_points.ply"
    _write_ply(init_ply, points[:: max(1, len(points) // 50000)])

    cameras = []
    for i, frame in enumerate(frames):
        cameras.append(
            {
                "name": frame.name,
                "extrinsic": extrinsics[i].tolist(),
                "intrinsic": intrinsics[i].tolist(),
            }
        )
    cameras_json = colmap_root / "cameras.json"
    cameras_json.write_text(json.dumps({"cameras": cameras}, indent=2))
    return init_ply, cameras_json


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to COLMAP quaternion (w, x, y, z)."""
    m = R
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def _write_colmap_text_fallback(
    frames: list[Path],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    points: np.ndarray,
    colmap_root: Path,
    images_dir: Path,
    sparse_dir: Path,
) -> tuple[Path, Path]:
    """Minimal COLMAP text export when pycolmap is unavailable."""
    for frame in frames:
        dst = images_dir / frame.name
        if not dst.exists():
            shutil.copy2(frame, dst)

    cameras_txt = sparse_dir / "cameras.txt"
    images_txt = sparse_dir / "images.txt"
    points_txt = sparse_dir / "points3D.txt"

    from PIL import Image

    with open(cameras_txt, "w") as f:
        f.write("# Camera list\n")
        for i, frame in enumerate(frames):
            with Image.open(frame) as im:
                w, h = im.size
            K = intrinsics[i]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            cid = i + 1
            f.write(f"{cid} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

    with open(images_txt, "w") as f:
        f.write("# Image list\n")
        for i, frame in enumerate(frames):
            ext = extrinsics[i]
            R = ext[:3, :3]
            t = ext[:3, 3]
            q = _rot_to_quat(R)
            iid = i + 1
            cid = i + 1
            name = frame.name
            f.write(f"{iid} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {cid} {name}\n\n")

    with open(points_txt, "w") as f:
        f.write("# 3D point list\n")
        step = max(1, len(points) // 50000)
        for pid, pt in enumerate(points[::step], start=1):
            f.write(f"{pid} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} 128 128 128 0.0\n")

    init_ply = colmap_root / "init_points.ply"
    _write_ply(init_ply, points[:: max(1, len(points) // 50000)])

    cameras_json = colmap_root / "cameras.json"
    cameras = [
        {"name": f.name, "extrinsic": extrinsics[i].tolist(), "intrinsic": intrinsics[i].tolist()}
        for i, f in enumerate(frames)
    ]
    cameras_json.write_text(json.dumps({"cameras": cameras}, indent=2))
    return init_ply, cameras_json
