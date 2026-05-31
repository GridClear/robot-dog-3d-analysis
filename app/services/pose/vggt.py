"""VGGT feed-forward pose + COLMAP dataset export (proven on DGX Spark GB10)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.services.dataset.colmap_layout import write_colmap_from_vggt
from app.services.pose.base import PoseResult

_MODEL = None


def _load_model(device: str):
    global _MODEL
    if _MODEL is None:
        from vggt.models.vggt import VGGT

        _MODEL = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    return _MODEL


class VGGTPoseBackend:
    name = "vggt"

    def is_available(self) -> tuple[bool, str]:
        try:
            import torch
            from vggt.models.vggt import VGGT  # noqa: F401

            cuda = torch.cuda.is_available()
            return True, f"torch {torch.__version__}, cuda={cuda}"
        except Exception as e:
            return False, str(e)

    def estimate(self, frames: list[Path], work_dir: Path) -> PoseResult:
        n = len(frames)
        try:
            import torch
            from vggt.utils.load_fn import load_and_preprocess_images
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        except Exception as e:
            return PoseResult(False, self.name, n, note=f"backend unavailable: {e}")

        work_dir.mkdir(parents=True, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        try:
            model = _load_model(device)
            images = load_and_preprocess_images([str(p) for p in frames]).to(device)
            _, _, h, w = images.shape
            with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], dtype=dtype):
                preds = model(images)

            pose_enc = preds["pose_enc"]
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                pose_enc, image_size_hw=(h, w), build_intrinsics=True
            )
            extrinsics = extrinsics.squeeze(0).float().cpu().numpy()
            intrinsics = intrinsics.squeeze(0).float().cpu().numpy()

            world = preds["world_points"].squeeze(0).float().cpu().numpy().reshape(-1, 3)
            conf = preds.get("world_points_conf")
            if conf is not None:
                mask = conf.squeeze(0).float().cpu().numpy().reshape(-1) > 0.5
                world = world[mask]

            colmap_root = work_dir / "colmap"
            init_ply, cameras_json = write_colmap_from_vggt(
                frames, extrinsics, intrinsics, world, colmap_root
            )
            return PoseResult(
                available=True,
                method=self.name,
                n_views=n,
                colmap_dir=colmap_root,
                cameras_json=cameras_json,
                init_points_ply=init_ply,
                note=f"{int(world.shape[0])} init points",
            )
        except Exception as e:
            return PoseResult(False, self.name, n, note=f"pose failed: {e}")
