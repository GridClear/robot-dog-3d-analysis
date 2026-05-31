# Robot-Dog World Model → Gaussian Splat

Stream a colored 3D Gaussian-splat reconstruction of a scene from a single
robot-dog camera frame. Drop one or more photos into the browser, type a
prompt, pick a camera motion, and the service plays each image forward
through the SANA-WM interactive world model to imagine new viewpoints,
then lifts those imagined frames into a real, orbitable 3D scene that
shows up live in the browser.

```
                 (image + prompt + WASD action)
                              │
                              ▼
        ┌──────────────────────────────────────────────┐
        │ SANA-WM (NVlabs/Sana)                        │
        │   image-to-video diffusion DiT (1.6B)        │
        │   + LTX-2 refiner (optional)                 │
        │   camera-controlled, 704×1280 @ 16 fps       │
        └──────────────────────────────────────────────┘
                              │  posed mp4 clip
                              ▼
        ┌──────────────────────────────────────────────┐
        │ Pi3X (yyfz233/Pi3X)                          │
        │   permutation-equivariant feed-forward       │
        │   multi-view geometry network                │
        │   → fused world-space point + confidence map │
        └──────────────────────────────────────────────┘
                              │  N×H×W×3 points + RGB
                              ▼
        ┌──────────────────────────────────────────────┐
        │ Splat packer (this repo)                     │
        │   conf-mask + random downsample              │
        │   OpenCV → WebGL axis convention             │
        │   32 bytes/gaussian Niantic .splat binary    │
        └──────────────────────────────────────────────┘
                              │  latest.splat (no-cache)
                              ▼
        ┌──────────────────────────────────────────────┐
        │ Browser (gsplat WebGL viewer)                │
        │   live SSE updates · orbit / zoom / pan      │
        └──────────────────────────────────────────────┘
```

The service runs as a long-lived FastAPI process and queues one image at
a time on the GPU. As soon as a generation finishes, every connected
browser receives an SSE event with the new splat's URL and gaussian
count; the canvas swaps in the new cloud without a page reload.

---

## Why a Gaussian splat instead of a video clip?

A video clip locks the viewer to whatever camera path SANA-WM rolled
out. A Gaussian splat is a real 3D representation — millions of small
oriented blobs in world space — so the operator can freely orbit the
scene, look behind objects the camera passed, and stitch multiple
generations into one shared 3D map. Splats also render at hundreds of
FPS on commodity WebGL, so an SO-100 or Unitree-style telepresence
operator can move their viewpoint independently of the world-model
sampling cadence (which is typically tens of seconds per clip).

This repo uses the [Niantic / antimatter15](https://github.com/antimatter15/splat)
`.splat` container (32 bytes/gaussian, no spherical harmonics) because
it loads in a single fetch in every major WebGL splat viewer
(`gsplat`, `mkkellogg/GaussianSplats3D`, `antimatter15/splat`) and
ships well over the wire — `gzip` shaves another ~20 %.

---

## Hardware & software requirements

| Component        | Required                                                  |
| ---------------- | --------------------------------------------------------- |
| OS               | Linux (tested on Jetson-class GB10 / x86_64 with CUDA)    |
| GPU              | ≥ 24 GB VRAM (Sana DiT + LTX-2 refiner offload-friendly)  |
| Python (API)     | 3.12+                                                     |
| Conda (SANA env) | Miniconda / Mambaforge; one env named `sana`              |
| Disk             | ~ 14 GB for Sana DiT, ~ 85 GB extra if you enable LTX-2,  |
|                  | ~ 4 GB for Pi3X weights                                   |
| Network          | First-run HuggingFace downloads (`Efficient-Large-Model/  |
|                  | SANA-WM_bidirectional`, `yyfz233/Pi3X`)                   |

---

## Quickstart

```bash
# 1. API venv (FastAPI only — keeps API deps separate from SANA's torch stack)
./scripts/install.sh

# 2. Clone NVlabs/Sana into third_party/Sana and build the `sana` conda env
./scripts/setup_sana_wm.sh

# 3. (Optional) benchmark on this box; writes data/benchmark.json which the
#    API auto-picks up as runtime overrides for interval / steps / refiner
./scripts/benchmark_sana_wm.sh

# 4. Run the service
cp .env.example .env       # tweak WM_* knobs as desired
./scripts/run.sh           # → http://localhost:8090
```

Open `http://localhost:8090`, drop one or more images, type a prompt
(or leave blank to use the default robot-dog explorer prompt), pick a
camera-motion preset, and hit **Start stream**. The viewer initialises
empty; the first splat lands after one full SANA-WM + Pi3X pass
(typically 20–90 s depending on `WM_INFERENCE_STEP` and refiner state).

> Pi3X downloads ~3.8 GB on first call. Watch `~/.cache/huggingface/hub`
> if your first splat is taking unusually long.

---

## How it works

### 1. Camera-controlled video generation (SANA-WM)

`app/services/sana_runner.py` spawns the upstream
[`inference_sana_wm.py`](third_party/Sana/inference_video_scripts/inference_sana_wm.py)
inside the `sana` conda env. We pass the seed image, the prompt text,
the rolled-out WASD/IJKL action string (`app/motion_presets.py`), the
sampling-step count, and intrinsics. SANA-WM samples a latent video
with its 1.6 B DiT, conditions every chunk on the camera trajectory via
plücker raymaps, and decodes either with the LTX-2 refiner (high
fidelity, ~ 85 GB checkpoint) or the bare Sana VAE (`--no_refiner`,
fast). Output: a `704×1280` mp4 in a per-generation `work_NNNN/` dir.

### 2. Multi-view 3D lifting (Pi3X)

`scripts/build_splat.py` runs in the same conda env right after the
mp4 is produced. It samples `WM_SPLAT_NUM_VIEWS` evenly-spaced frames
from the clip, optionally prepends the original seed image as an
anchor view, and feeds them all to Pi3X (`yyfz233/Pi3X`) as a single
permutation-equivariant batch.

Pi3X returns four tensors per call:

| Key            | Shape                  | Meaning                                |
| -------------- | ---------------------- | -------------------------------------- |
| `points`       | `(B, N, H, W, 3)`      | World-space point map (already fused)  |
| `local_points` | `(B, N, H, W, 3)`      | Per-view camera-local points           |
| `conf`         | `(B, N, H, W, 1)`      | Confidence logits (sigmoid → `[0,1]`)  |
| `camera_poses` | `(B, N, 4, 4)`         | C2W matrices (OpenCV convention)       |

We take `points` directly because it's the multi-view-consistent
fused cloud; the SANA-WM action trajectory is not needed at this
stage.

### 3. Splat packing

Points with `sigmoid(conf) ≥ WM_SPLAT_CONF_THRESHOLD` are kept; if
nothing survives the threshold we fall back to the top-10 % by
confidence so the browser never sees an empty file. The cloud is
randomly downsampled to `WM_SPLAT_MAX_GAUSSIANS` (default 250 k →
~ 8 MB), then converted from OpenCV (Y-down, Z-forward) to WebGL
(Y-up, Z-back) by flipping `y` and `z`. Every gaussian gets the same
scene-adaptive isotropic scale (`0.1 %` of the bounding-box diagonal)
and an identity quaternion; per-point RGB comes straight from the
matching Pi3X-input pixel.

Each record is 32 bytes, packed exactly per
[Polyvia3D's spec](https://polyvia3d.com/formats/splat):

```
bytes  0–11 : float32 position (x, y, z)
bytes 12–23 : float32 scale    (sx, sy, sz)
bytes 24–27 : uint8   RGBA     (r, g, b, a=255)
bytes 28–31 : uint8   quat     ((q*128+128) clamped to [0,255])
```

A sibling `.splat.json` records `n_gaussians`, the chosen scale, and
the source mp4 for debugging.

### 4. Browser delivery

`app/routers/sessions.py` exposes one streaming endpoint per session
(`GET /v1/sessions/{id}/events`, SSE) and one artifact endpoint
(`GET /v1/sessions/{id}/splat/latest.splat`). The static page in
`static/index.html` loads `gsplat` from a public CDN and rebuilds the
scene every time it receives a `generation_ready` event:

```js
import * as SPLAT from "https://cdn.jsdelivr.net/npm/gsplat@latest/+esm";

state.scene = new SPLAT.Scene();
await SPLAT.Loader.LoadAsync(`${url}?v=${generation_id}`, state.scene);
```

`?v=<generation_id>` busts any intermediate cache; the server also
sets `Cache-Control: no-store` on the splat endpoint.

---

## API reference

| Method | Path                                      | Purpose                                      |
| ------ | ----------------------------------------- | -------------------------------------------- |
| GET    | `/healthz`                                | SANA readiness + GPU summary                 |
| GET    | `/v1/sessions/config`                     | Runtime knobs + motion presets               |
| POST   | `/v1/sessions`                            | Multipart upload of one or more images       |
| GET    | `/v1/sessions/{id}`                       | Session record + per-image queue state       |
| POST   | `/v1/sessions/{id}/start`                 | Kick off background generation loop          |
| POST   | `/v1/sessions/{id}/stop`                  | Cancel after the current generation          |
| GET    | `/v1/sessions/{id}/events`                | SSE: `generation_started/ready/failed`, etc. |
| GET    | `/v1/sessions/{id}/splat/latest.splat`    | Most recent `.splat` (32 B / gaussian)       |

SSE `generation_ready` payload:

```json
{
  "type": "generation_ready",
  "session_id": "ab12cd34ef56",
  "index": 0,
  "generation_id": 1,
  "url": "/v1/sessions/ab12cd34ef56/splat/latest.splat",
  "n_gaussians": 248317,
  "elapsed_sec": 41.7,
  "next_in_sec": 18.3
}
```

---

## Repository layout

```
app/
  config.py            Settings (env_prefix=WM_, .env-aware)
  main.py              FastAPI app + static mount
  motion_presets.py    WASD/IJKL DSL presets (forward_explore, orbit_left…)
  routers/
    health.py          /healthz
    sessions.py        /v1/sessions/*
  schemas.py           Pydantic models for the queue + responses
  services/
    sana_runner.py     conda subprocess: SANA-WM → mp4 → Pi3X → .splat
    storage.py         per-session dirs, metadata I/O, image normalisation
    stream_worker.py   one-at-a-time GPU queue + SSE pub/sub
scripts/
  build_splat.py       Pi3X feed-forward + 32B-record packer (run in `sana`)
  setup_sana_wm.sh     clone NVlabs/Sana + build conda env
  install.sh           API venv
  benchmark_sana_wm.sh write data/benchmark.json (interval / steps / refiner)
  run.sh               uvicorn launcher
static/
  index.html           single-page UI: drop images → orbit splat
tests/
  test_sessions_api.py mocked end-to-end (no GPU)
  test_live_wm.py      gated by WM_LIVE=1
third_party/Sana/      cloned by setup_sana_wm.sh (gitignored)
```

---

## Configuration (`.env` / env vars)

All knobs read from `WM_*` env vars; the API auto-applies overrides
from `data/benchmark.json` when present.

| Variable                  | Default                                  | Notes                                          |
| ------------------------- | ---------------------------------------- | ---------------------------------------------- |
| `WM_HOST` / `WM_PORT`     | `0.0.0.0` / `8090`                       |                                                |
| `WM_DATA_DIR`             | `data`                                   | Per-session subdirs go in `data/sessions/`     |
| `WM_INTERVAL_SEC`         | `30`                                     | Minimum gap between generations                |
| `WM_NUM_FRAMES`           | `321`                                    | Capped to `≤ WM_MAX_CLIP_SECONDS * fps`        |
| `WM_FPS`                  | `16`                                     |                                                |
| `WM_USE_REFINER`          | `false`                                  | Enables the 85 GB LTX-2 refiner               |
| `WM_INFERENCE_STEP`       | `60`                                     | DiT sampling steps                             |
| `WM_OFFLOAD_VAE`          | `false`                                  | VAE → CPU between encode/decode                |
| `WM_OFFLOAD_REFINER`      | `false`                                  | Lazy-load LTX-2 between calls                  |
| `WM_NVFP4_ENABLED`        | `false`                                  | Enables `--nvfp4` once upstream ships it       |
| `WM_ACTION_PRESET`        | `forward_explore`                        | One of `app/motion_presets.py`                 |
| `WM_DEFAULT_INTRINSICS`   | demo                                     | Skips Pi3X intrinsics estimation               |
| `WM_SPLAT_NUM_VIEWS`      | `8`                                      | Frames sent to Pi3X per generation             |
| `WM_SPLAT_MAX_GAUSSIANS`  | `250000`                                 | Random downsample target (~ 8 MB)              |
| `WM_SPLAT_CONF_THRESHOLD` | `0.30`                                   | `sigmoid(conf)` cutoff                         |
| `WM_SPLAT_TIMEOUT_SEC`    | `600`                                    | Hard kill if Pi3X stalls                       |

---

## Testing

```bash
./.venv/bin/python -m pytest -q          # mocked, no GPU
WM_LIVE=1 ./.venv/bin/python -m pytest   # exercises real SANA + Pi3X
```

---

## Limitations & roadmap

- **No spherical harmonics.** The `.splat` container drops view-dependent
  lighting by design. Specular highlights baked into SANA-WM frames will
  look the same from every angle.
- **Per-clip splats are independent.** Successive generations replace
  the scene; we don't yet fuse them into a persistent map. A future
  pass can register Pi3X clouds across generations (ICP or Procrustes
  against SANA-WM's known camera trajectory).
- **No true 3DGS training.** What we ship is a Pi3X-lifted colored
  point cloud packaged in the Gaussian-splat wire format. Real 3DGS
  optimisation (Plenoxels / `gsplat` training) would give better
  density and surface coverage at the cost of minutes per generation.
- **Pi3X warm-up cost.** Each generation re-loads Pi3X (~ 10 s warm,
  much longer cold). A long-lived Pi3X daemon would amortise this.
- **One GPU only.** `stream_worker` serialises generations behind an
  `asyncio.Lock`; multi-GPU sharding is straightforward but unimplemented.

---

## Acknowledgements

- [NVlabs/Sana](https://github.com/NVlabs/Sana) — the SANA-WM
  bidirectional checkpoint, LTX-2 refiner, and inference script we
  wrap.
- [yyfz/Pi3 (Pi3X)](https://github.com/yyfz/Pi3) — the
  permutation-equivariant feed-forward 3D geometry model we use to
  lift video frames.
- [antimatter15/splat](https://github.com/antimatter15/splat) and
  [huggingface/gsplat](https://github.com/huggingface/gsplat) — for
  the 32-byte wire format and the WebGL viewer this UI depends on.
