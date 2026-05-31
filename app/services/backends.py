"""Backend registry and GPU status."""
from __future__ import annotations

from typing import Any

from app.schemas import BackendStatus, BackendsStatusResponse
from app.services.pose.mast3r import MASt3RPoseBackend
from app.services.pose.vggt import VGGTPoseBackend
from app.services.train.gsplat import GSplatTrainBackend


def get_pose_backend(name: str):
    backends = {
        "vggt": VGGTPoseBackend(),
        "mast3r": MASt3RPoseBackend(),
    }
    if name not in backends:
        raise ValueError(f"unknown pose backend: {name}")
    return backends[name]


def get_train_backend(name: str):
    backends = {
        "gsplat": GSplatTrainBackend(),
    }
    if name not in backends:
        raise ValueError(f"unknown train backend: {name}")
    return backends[name]


def backends_status() -> BackendsStatusResponse:
    pose = []
    for cls in (VGGTPoseBackend, MASt3RPoseBackend):
        b = cls()
        ready, note = b.is_available()
        pose.append(BackendStatus(name=b.name, ready=ready, note=note))

    train = []
    for cls in (GSplatTrainBackend,):
        b = cls()
        ready, note = b.is_available()
        train.append(BackendStatus(name=b.name, ready=ready, note=note))

    return BackendsStatusResponse(pose=pose, train=train, gpu=_gpu_info())


def _gpu_info() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        free, total = torch.cuda.mem_get_info(0)
        return {
            "cuda": True,
            "device": torch.cuda.get_device_name(0),
            "memory_free_gb": round(free / 1e9, 2),
            "memory_total_gb": round(total / 1e9, 2),
        }
    except Exception as e:
        return {"cuda": False, "error": str(e)}
