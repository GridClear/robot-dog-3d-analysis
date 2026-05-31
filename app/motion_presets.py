"""WASD/IJKL action strings for SANA-WM camera control."""
from __future__ import annotations

PRESETS: dict[str, str] = {
    "forward_explore": "w-100,w-80,w-80,w-60",
    "forward_slow": "w-60,w-60,w-60,w-60",
    "pan_left": "w-40,j-40,w-40,j-40,w-40",
    "pan_right": "w-40,l-40,w-40,l-40,w-40",
    "orbit_left": "jw-50,w-50,jw-50,w-50",
    "hold": "none-160",
}


def resolve_action(preset_or_dsl: str) -> str:
    key = preset_or_dsl.strip()
    if key in PRESETS:
        return PRESETS[key]
    return key
