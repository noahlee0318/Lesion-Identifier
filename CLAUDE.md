# CLAUDE.md — Lesion Atlas

Read this before doing anything. It carries decisions that are already locked.
Full reasoning lives in `docs/build-plan.md`, `docs/runbook.md`, and
`docs/capture-geometry.md`.

## What this is

A personal longitudinal acne-tracking system. Daily 3-pose facial capture ->
detect acne lesions across 6 face regions -> hold lesion identity across days
in a fixed anatomical coordinate frame -> visualize count and spatial
distribution change over months.

Single subject (Noah). Runs entirely on a Windows 11 Lenovo LOQ with an
NVIDIA GPU. Capture device is an iPhone 15. Nothing leaves the laptop.

**Current state: phase 0 tooling built and verified against synthetic
fixtures. Blocked on real calibration photos.** See `README.md` for commands
and `docs/PHASE0_CHECKLIST.md` for the physical steps Noah has to do.

## Locked decisions — do NOT re-litigate these

These were argued through and settled on 2026-09-11. If you think one is
wrong, say so once in one paragraph and then proceed as specified. Do not
silently redesign around them.

- **3 poses: frontal, left 60 deg yaw, right 60 deg yaw.** No zooming.
- **6 regions**: forehead, chin & perioral, L cheek (to jawline), R cheek
  (to jawline), L temple & sideburn, R temple & sideburn.
- **Region ownership**: each region is counted in exactly one pose.
  Frontal owns forehead + chin/perioral. Each 60 deg pose owns its cheek +
  temple/sideburn. Detections outside an owning pose are kept with
  `owned=false` for cross-pose validation only. This is what prevents
  double-counting the mid-cheek overlap band.
- **Three atlases, one per pose, frozen forever** once chosen.
- **One detector across all poses**, with pose as a feature. Not three models.
- **Labeling budget: 300-600 images** over a few weeks.
- **v1 scope is phases 0-4.** The detector is in v1 but cannot be built
  first; it needs labels that do not exist yet.

### Why 60 deg (so you do not re-derive it)

Skin at angle theta to the sensor projects at cos(theta) of true size.
Incidence ~= |phi - yaw|, where phi is a face point's azimuth from
straight-ahead (nose ~0 deg, directly lateral ~90 deg).

| Region | phi | Frontal | 45 deg | **60 deg** | 90 deg |
|---|---|---|---|---|---|
| Forehead, center | ~5 | 5 | 40 | 55 | 85 |
| Medial cheek | ~30 | 30 | 15 | 30 | 60 |
| Lateral cheek | ~60 | 60 | 15 | **0** | 30 |
| Sideburn / preauricular | ~80 | 80 | 35 | **20** | 10 |

- **Zoom does not fix foreshortening.** A tighter crop shows the same angle,
  bigger. Pose is the only lever. Do not propose optical zoom as a fix.
- **Not 90 deg profiles**: Face Mesh breaks past ~+/-60 deg yaw
  (self-occlusion), and frontal/profile share almost no skin, so the
  load-bearing mid-cheek overlap band (phi ~30-50 deg) disappears. That band
  ties the three pose frames into one head frame and yields a daily
  self-reporting registration-error estimate.
- **Not 45 deg**: 60 deg drops the sideburn from 35 to 20 deg incidence and
  puts the lateral cheek at 0, while staying inside Face Mesh's range.
- **The one open geometry question**: if phase 0 shows Face Mesh does not
  fire reliably at 60 deg, step down to 50 deg. That constraint sets the
  angle, not aesthetics.

## Gates — do not advance a phase until its gate passes

| Phase | Gate |
|---|---|
| P0 spike | Registration median err < 1.5mm, p95 < 3mm; angle locked |
| P1 capture app | Session under 60s, 7 consecutive days running |
| P2 labeling | Self-agreement > ~0.75 on 20 re-labeled images |
| P4 detector | Precision > 0.8 at Recall > 0.7 on held-out **days** |

## Phase map

0. **Spike** (wk1) — Face Mesh at 60 deg; registration method bake-off;
   lens test; ingest server; iOS verification tests.
1. **Capture rig + app** (wk1-2) — HTTPS, getUserMedia, 3-step wizard,
   ghost overlay at 35% opacity, live alignment readout in mm.
2. **Labeling** (wk2-6) — RUBRIC.md, custom canvas tool, active learning.
3. **Canonical space** (wk3-4) — freeze atlases, register, region polygons.
4. **Detector** (wk5-7) — 640px tiles @ 20% overlap, sliced inference, YOLO
   or RT-DETR.
5. **Tracking** (wk7-8) — Hungarian matching in canonical mm, 4mm gate.
6. **Visualization** (wk8+) — counts over time, density heatmaps, swimlanes.
7. **Later ML** — DINOv2 self-supervised pretraining, resolution forecasting.

Daily capture is not a phase. It starts on day 3 of week 1 and never stops.

## Hard rules

- **Millimeters, never pixels**, in anything a human reads. Convert via the
  per-image ArUco px/mm, not a global constant.
- **Never re-encode images.** The server stores received bytes unmodified —
  no resize, no EXIF strip, no "normalization". Store a SHA-256 per file.
- **Reject, do not average.** A bad registration is discarded, never blended
  into an atlas.
- **Split by date, never randomly.** Consecutive days are near-duplicates;
  a random split leaks badly and will produce fake metrics.
- **Control points are held-out ground truth.** Never fit a transform using
  them. If you are tempted, stop and say so.
- **Log covariates from day one.** They cannot be collected retroactively.
- **Adherence is the #1 project risk.** Capture must stay under 60s. Every
  added pose is daily friction. Weigh features against that.
- **Gen AI has near-zero value for the core task.** Legit uses: VLM
  pre-labeling in P2, a natural-language query layer. Diffusion synthetic
  data will likely hurt on a single-subject dataset. Do not propose it.

## Device constraints (iPhone 15 / iOS Safari)

- **`ImageCapture` is unsupported in Safari.** It is Chromium-only. Do not
  write `ImageCapture.takePhoto()`. See `docs/runbook.md` for the two
  fallbacks and the P0 measurement that decides between them.
- **iOS Safari will not honor manual exposure/WB constraints.** Call
  `applyConstraints` anyway and **log what was actually granted** rather
  than assuming it worked.
- **A file input needs no secure context.** Plain HTTP is fine for the P0
  upload page. HTTPS (mkcert or Tailscale) is required only for
  `getUserMedia` in P1.
- Camera set to **Most Compatible** (writes JPEG, not HEIC) and Photos set
  to **Keep Originals**.
- Safari's file picker may alter uploads. This is verified in P0 before any
  image joins the longitudinal series.

## Conventions

- Images: `data/raw/YYYY-MM-DD/<pose>_<lens>_<device>_<HHMMSS>.jpg`
- Poses: `frontal` | `left60` | `right60`
- Lenses: `main` | `2x`
- JPEG q~92. Never HEIC, never raw.
- `data/` and `reports/` are gitignored and never committed.
- Schema: `sessions` -> `images` (keyed by pose + lens) -> `detections`
  (per-image, disposable, regenerable; carries `region` + `owned`) and
  `lesions` (durable cross-day identity), joined by `observations`. Plus
  `labels` and a dated `regimen_events` log.
- Fiducial: ArUco `DICT_4X4_50`, id 0, 30mm, matte paper, plus color patch.

## Environment

- Windows 11, Lenovo LOQ, RTX 4050 Laptop (6 GiB), driver 591.66. Verified
  2026-09-11: torch 2.11.0+cu128, `torch.cuda.is_available()` True. Say so
  plainly if it stops being true — do not silently fall back to CPU.
- **Python 3.12 is required.** The machine's default `python` is 3.14, and
  mediapipe publishes no 3.13/3.14 wheels. The venv is built from the
  uv-managed 3.12.13 interpreter. `scripts\verify_env.py` checks this first.
- **mediapipe is 1.0.1, which REMOVED `mp.solutions` entirely.**
  `FaceMesh(refine_landmarks=True)` no longer exists. Use the Tasks API
  `FaceLandmarker` with `models/weights/face_landmarker.task` — it emits the
  same 478 landmarks (468 face + 10 iris) with the same indices, plus a
  facial transformation matrix that gives head pose directly. See
  `src/facemesh_check.py`.
- **OpenCV is 5.0.** ArUco, SIFT, ORB and the TPS shape transformer are all
  present in the contrib build; the `ArucoDetector` class API is used, not the
  removed free-function `detectMarkers`.
- 6 GiB of VRAM is the real constraint on LoFTR. It runs at a capped
  resolution and matches are scaled back to full res — see the caps at the top
  of `src/register.py`.
- `opencv-contrib-python` (contrib, for ArUco — never alongside plain
  `opencv-python`), mediapipe, numpy, scipy, matplotlib, torch+torchvision
  CUDA, kornia, pillow.
- Pre-fetch LightGlue/DISK and LoFTR weights and cache them in-repo
  (gitignored) so the spike cannot fail on a download.
- **Ask before adding any dependency not already in `requirements.txt`.**

## Privacy

The photos are several hundred high-resolution images of Noah's face.

- **The repo contains NO images, and is configured so it cannot.**
  `.gitignore` blanket-bans every raster and video format with no exceptions
  carved out. Nothing the project displays is committed; it is generated by a
  command or fetched by a script.
- **`DATA_ROOT` lives OUTSIDE the repo** — `C:\LesionAtlas\data` by default,
  override with `LESION_ATLAS_DATA`. Decided 2026-09-11: the repo folder sits
  in `OneDrive\Desktop`, and OneDrive has no reliable per-subfolder upload
  exclusion, so keeping `data/` off the synced tree is the only way to honour
  the no-cloud rule. **Do not move the data root back inside the repo.**
- **A git remote exists, and it is code-only.** Decided 2026-09-11 at Noah's
  request, deliberately before any real image existed. This supersedes the
  earlier "no git remote" rule for CODE. It does not relax anything about
  images: no photo is ever committed, force-added, or pushed.
- Backup goes to an external drive or local folder, never a git host and never
  a cloud service. `scripts\backup_data.ps1`.
- No image is ever sent to a third-party API. Not for labeling, not for
  inference, not for "just checking something".

## How Noah wants work run

- **Blunt over encouraging.** If a result is bad, say it is bad. If an idea
  is wrong, say so before he spends a week on it. No praise padding.
- State the verdict first, then the reasoning.
- Build in reviewable chunks and verify each one before moving on.
- When a gate fails, name what failed and the cheapest next lever. Do not
  soften it.
- Flag the physical-world steps he has to do himself rather than working
  around them — rigging, printing, shooting. You cannot do those.

## Open questions

- Face Mesh reliability at 60 deg (P0 decides; fallback 50 deg).
- Which registration method wins (P0 bake-off).
- Main lens @ 30cm vs 2x @ 55cm (P0 lens test).
- Does iOS Safari alter uploaded files (P0 fidelity check).
- px/mm at video-stream resolution — decides whether P1 keeps the ghost
  overlay or switches to native capture.
- ACNE04 dataset availability and license (P4, check before relying on it).
- The six region polygons are named but not yet defined in canonical
  coordinates (P3).
