# CLAUDE.md — Lesion Atlas

Read this before doing anything. It carries decisions that are already locked.

- `docs/DECISIONS.md` — dated log of settled decisions and why. **Authoritative
  where anything else disagrees.**
- `docs/build-plan.md`, `docs/runbook.md`, `docs/capture-geometry.md` — the
  11 Sept morning snapshots, converted from the artifact PDFs. Rich on
  reasoning, but they predate several decisions; DECISIONS.md supersedes them.
- `README.md` — commands. `docs/PHASE0_CHECKLIST.md` — the physical steps Noah
  does himself.

## What this is

A personal longitudinal acne-tracking system. Daily 3-pose facial capture ->
detect acne lesions across 6 face regions -> hold lesion identity across days
in a fixed anatomical coordinate frame -> visualize count and spatial
distribution change over months.

Single subject (Noah). Runs entirely on a Windows 11 Lenovo LOQ with an
NVIDIA GPU. Capture device is an iPhone 15. Nothing leaves the laptop.

**Current state: phase 0 tooling built and verified against synthetic
fixtures. First real photos arrived 2026-09-11 (2 calibration shots, both
failed QA). Blocked on real calibration photos — five left-60 shots across three
calendar days, which cannot be compressed.** The gap-work items (tiler,
date split) are built; the labeling tool is not started.

## What is built

Verified against synthetic fixtures only. No real photo has entered any of it.

**Phase 0 spike**
| File | Does | State |
|---|---|---|
| `src/fiducial.py` | ArUco + colour patch target, print verification, px/mm | built |
| `src/facemesh_check.py` | Landmark detection, head pose, per-landmark stability, angle verdict | built, awaiting photos |
| `src/controlpoints.py` | Hand-marking tool, stable mole ids across images | built |
| `src/register.py` | Method-ladder bake-off, pooled mm residuals, TPS, blink review, GO/NO-GO | built, awaiting photos |
| `src/lens_compare.py` | Per-region sharpness, main @30cm vs 2x @55cm | built, awaiting photos |

**Ingest (P0 half of the P1 server — deliberately no camera code)**
| File | Does | State |
|---|---|---|
| `src/server/main.py` | FastAPI, plain HTTP, `/`, `/capture`, `/health`, `/probe` | built |
| `src/server/db.py` | SQLite: sessions, images, regimen_events | built |
| `src/server/qa.py` | Blur, exposure, fiducial-found, px/mm per upload; plain-English rejections | built, thresholds provisional |
| `src/server/static/index.html` | Upload page, covariate taps, regimen form | built |
| `src/server/static/probe.html` | getUserMedia resolution + granted-constraints probe | built, awaiting run |

**Pipeline foundations (gap work)**
| File | Does | State |
|---|---|---|
| `src/tiling.py` | 640px/20% grid, image<->tile coords, membership, visibility | built, tested |
| `src/splits.py` | Date-based splits, trailing windows, washout, write-once | built, tested |
| `src/config.py` | Paths, locked domain constants, gate thresholds | built |

**Scripts**: `verify_env.py`, `fetch_weights.py`, `install_deps.ps1`,
`install_autostart.ps1` / `uninstall_autostart.ps1`, `backup_data.ps1`,
`verify_upload_fidelity.py`.

**Tests**: `test_tiling.py`, `test_splits.py`, `test_register.py`,
`test_qa_messages.py`, `make_synthetic.py` (fixture generator). All run
standalone (`python -m tests.test_x`) — pytest is not installed.

**Not built**: the labeling tool (P2), the capture app (P1 — blocked on the
`/probe` measurement), atlas freezing and `regions.py` (P3), detector (P4),
tracking (P5), visualization (P6).

## Locked decisions — do NOT re-litigate these

Settled 2026-09-11. If you think one is wrong, say so once in one paragraph and
then proceed as specified. Do not silently redesign around them.

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
Incidence ~= |phi - yaw|, phi being azimuth from straight-ahead (nose ~0 deg,
directly lateral ~90 deg).

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
- **The one open geometry question**: if phase 0 shows Face Mesh does not fire
  reliably at 60 deg, step down to 50 deg. That constraint sets the angle, not
  aesthetics.

## Gates — do not advance a phase until its gate passes

| Phase | Gate |
|---|---|
| P0 spike | Registration **pooled** median < 1.5mm, p95 < 3mm; angle locked |
| P1 capture app | Session under 60s, 7 consecutive days running |
| P2 labeling | Self-agreement > ~0.75 on 20 re-labeled images |
| P4 detector | Precision > 0.8 at Recall > 0.7 on held-out **days** |

The P0 registration gate reads residuals **pooled across all pairs**, never a
median of per-pair medians — a per-pair median hides one catastrophic pair.
Only methods that scored on every pair are eligible. The recommendation is the
**simplest method that clears**, not the lowest number; pooled median breaks
ties inside a complexity tier and nowhere else. A p95 is refused below 20
pooled residuals, and a median that clears while p95 is unevaluable is
**INCONCLUSIVE**, not GO.

A pair that **no** method could score — usually fewer than three control-point
ids shared with the reference — is dropped before eligibility is judged and
named loudly above the verdict as a *data* problem. Without that, one
under-marked image makes every method incomplete, nothing is eligible, and the
headline reads NO-GO: a registration failure announced because of an
annotation. Dropping such a pair can only ever add eligible methods; a pair
scored by *some* methods stays in, and the methods that missed it stay
ineligible. Do not "simplify" this away.

**The blink compare must show the transform the millimetres score.** For a TPS
the dense warp inverts the spline numerically (Newton, seeded from a reverse
fit) and the review screen prints the measured round-trip error. A separately
fitted reverse spline is *not* an inverse, and showing one would defeat the
one check that catches a warp that is wrong while the numbers look fine.

**Only the angle lock gates real daily sessions.** The registration gate does
not. If registration comes back at 2.5mm, that is a software problem solved
with better matching — the photos are still valid and still belong to the
series. Do not hold up capture waiting on it.

## Phase map

0. **Spike** (wk1) — Face Mesh at 60 deg; registration bake-off; lens test;
   ingest server; iOS verification tests.
1. **Capture rig + app** (wk1-2) — HTTPS, getUserMedia, 3-step wizard,
   ghost overlay at 35% opacity, live alignment readout in mm.
2. **Labeling** (wk2-6) — RUBRIC.md, custom tool, active learning.
3. **Canonical space** (wk3-4) — freeze atlases, register, region polygons.
4. **Detector** (wk5-7) — 640px tiles @ 20% overlap, sliced inference, YOLO
   or RT-DETR.
5. **Tracking** (wk7-8) — Hungarian matching in canonical mm, 4mm gate.
6. **Visualization** (wk8+) — counts over time, density heatmaps, swimlanes.
7. **Later ML** — DINOv2 self-supervised pretraining, resolution forecasting.

Daily capture is not a phase. It starts on day 3-4 of week 1 and never stops.
RUBRIC.md belongs to P2, not P0 — written with real borderline crops in hand,
because a rubric drafted before seeing real high-res skin is made of guesses.

## Hard rules

- **Millimeters, never pixels**, in anything a human reads. Convert via the
  per-image ArUco px/mm, not a global constant.
- **Never re-encode images.** The server stores received bytes unmodified —
  no resize, no EXIF strip, no "normalization". Store a SHA-256 per file.
- **Reject, do not average.** A bad registration is discarded, never blended
  into an atlas.
- **Labels live in IMAGE coordinates, never tile coordinates.** Tiles are a
  view, generated on demand, never a storage format. Per-tile storage would
  mean changing `tile` or `overlap` later invalidates every label ever made.
  Retiling must stay free.
- **Two membership rules, and they are not interchangeable.**
  `labels_in_tile` returns every label centred in a tile that is at least
  `min_visible` visible — **for training**, and a label may appear in several
  tiles, which is correct: a crop that renders a lesion must carry its label.
  `assign_labels_to_tiles` gives exactly one owner — **for counting**. Same
  shape as `detections.owned` for the pose overlap band.
- **Visibility filtering rests on a size bound, and the bound is computable.**
  `labels_in_tile` drops a label the tile barely shows, which is only safe
  because a neighbouring tile shows it whole — and that holds only while
  lesion diameter <= `tile - stride`. `max_safe_lesion_mm()` gives it: 128 px,
  **4.9 mm at 26 px/mm** for the locked 640/20%. The labeling tool must warn
  past it (`lesion_size_warning()`). Note the default `min_visible=0.5` prunes
  corner slivers only — a disc whose centre is inside a tile always has >= half
  its area there, so pruning edge slivers needs a value above 0.5.
- **Edge tiles are clamped, never zero-padded.** Every frame would pad on the
  same two edges, so the border correlates with position in frame — exactly the
  spurious signal a single-subject dataset latches onto. Sliced inference must
  inherit the same clamped grid.
- **Split by date, never randomly.** Consecutive days are near-duplicates; a
  random split leaks badly and will produce fake metrics. Default rule is fixed
  trailing calendar windows (14d test, 14d val, 3d washout), so the held-out
  set is the same size forever and always current. Washout is measured in
  **calendar days**, not date-index. Split files are **write-once**:
  `save_split` refuses to overwrite, because a metric with no record of which
  days were held out is worthless.
- **Control points are held-out ground truth.** Never fit a transform using
  them — and never **select** with them either: picking which method of the
  ladder to report, or which matcher to spline, using control-point error is
  selection on the test set and biases the gate downward. If you are tempted,
  stop and say so.
- **Per-upload QA measures skin, not the fiducial.** The marker is white paper
  and black ink and is in every frame on purpose — at rig geometry it is ~12%
  of the photo and its white alone ~6.5%, against a 2% clipping limit. It is
  masked out before exposure is measured. Counting it would reject every
  correctly exposed photo and blame the lamp, which teaches Noah to override
  the verdict, after which the verdict catches nothing.
- **Log covariates from day one.** They cannot be collected retroactively.
  Six daily 3-option tap rows; regimen change is a dated `regimen_events` row,
  never a tap row, because it is a step function and will drive most of the
  variance in the series.
- **Adherence is the #1 project risk.** Capture must stay under 60s. The server
  autostarts at logon so starting it is never a remembered step. Every added
  pose is daily friction. Weigh features against that.
- **Gen AI has near-zero value for the core task.** Legit uses: VLM
  pre-labeling in P2, a natural-language query layer. Diffusion synthetic data
  will likely hurt on a single-subject dataset. Do not propose it.

## Device constraints (iPhone 15 / iOS Safari)

- **`ImageCapture` is unsupported in Safari.** It is Chromium-only. Do not
  write `ImageCapture.takePhoto()`. The two fallbacks — canvas grab off the
  video stream (keeps the ghost overlay, capped at video resolution) vs
  `<input capture="environment">` (full still resolution, loses the overlay) —
  are decided by the `/probe` px/mm measurement, not by argument.
- **iOS Safari will not honor manual exposure/WB constraints.** Call
  `applyConstraints` anyway and **log what was actually granted** rather than
  assuming it worked.
- **A file input needs no secure context.** Plain HTTP is fine for the P0
  upload page. HTTPS (mkcert or Tailscale) is required only for `getUserMedia`
  in P1.
- Camera set to **Most Compatible** (writes JPEG, not HEIC) and Photos set to
  **Keep Originals**.
- Safari's file picker may alter uploads. Verified by
  `scripts/verify_upload_fidelity.py` before any image joins the series.

## Conventions

- Data root is **outside the repo**: `C:\LesionAtlas\data`, override with
  `LESION_ATLAS_DATA`. See Privacy.
- Images: `<DATA_ROOT>/raw/YYYY-MM-DD/<pose>_<lens>_<device>_<HHMMSS>.jpg`
- Poses: `frontal` | `left60` | `right60`. Lenses: `main` | `2x`.
- JPEG q~92. Never HEIC, never raw.
- Schema: `sessions` -> `images` (keyed by pose + lens) -> `detections`
  (per-image, disposable, regenerable; carries `region` + `owned`) and
  `lesions` (durable cross-day identity), joined by `observations`. Plus
  `labels` and a dated `regimen_events` log.
- `sessions.kind` is `'session'` or `'calibration'`. Calibration images never
  enter a split, a training set, or a count.
- Tiles: 640px, 20% overlap, stride 512, ids `r{row}c{col}` derived from
  position. Membership rect is half-open `[x0,x1) x [y0,y1)`.
- Fiducial: ArUco `DICT_4X4_50`, id 0, 30mm, matte paper, plus colour patch.
- Locked domain constants live in `src/config.py`, not scattered as literals.

## Environment

- Windows 11, Lenovo LOQ, RTX 4050 Laptop (6 GiB), driver 591.66. Verified
  2026-09-11: torch 2.11.0+cu128, `torch.cuda.is_available()` True. Say so
  plainly if it stops being true — do not silently fall back to CPU.
- **Python 3.12 is required.** The machine's default `python` is 3.14, and
  mediapipe publishes no 3.13/3.14 wheels. The venv is built from the
  uv-managed 3.12.13 interpreter. `scripts\verify_env.py` checks this first.
- **mediapipe is 1.0.1, which REMOVED `mp.solutions` entirely.**
  `FaceMesh(refine_landmarks=True)` no longer exists. Use the Tasks API
  `FaceLandmarker` with `models/weights/face_landmarker.task` — same 478
  landmarks (468 face + 10 iris) with the same indices, plus a facial
  transformation matrix giving head pose directly. See `src/facemesh_check.py`.
- **OpenCV is 5.0.** ArUco, SIFT, ORB and the TPS shape transformer are all
  present in the contrib build; the `ArucoDetector` class API is used, not the
  removed free-function `detectMarkers`.
- 6 GiB of VRAM is the real constraint on LoFTR. It runs at a capped resolution
  and matches are scaled back to full res — see the caps at the top of
  `src/register.py`.
- `opencv-contrib-python` (contrib, for ArUco — never alongside plain
  `opencv-python`), mediapipe, numpy, scipy, matplotlib, torch+torchvision
  CUDA, kornia, pillow, fastapi, uvicorn.
- LightGlue/DISK and LoFTR weights are cached in-repo under `models/weights`
  (gitignored) by `scripts/fetch_weights.py`, so a spike cannot fail on a
  download.
- **Ask before adding any dependency not already in `requirements.txt`.**

## Privacy

The photos are several hundred high-resolution images of Noah's face.

- **The repo contains NO images, and is configured so it cannot.**
  `.gitignore` blanket-bans every raster and video format with no exceptions
  carved out. Nothing the project displays is committed; it is generated by a
  command or fetched by a script.
- **The whole project lives on a plain local path, off any cloud sync.**
  Decided 2026-09-11. Layout:
  ```
  C:\LesionAtlas\
      data\           <- DATA_ROOT, never committed
      lesion-atlas\   <- the repo
  ```
  It originally sat in `OneDrive\Desktop\pimple detector`, which would have
  auto-uploaded every photo to Microsoft. OneDrive has no reliable
  per-subfolder upload exclusion, so the project was moved out entirely rather
  than an exception carved for `data/`.
  **Do not move it back under OneDrive, Dropbox, iCloud or Google Drive.**
- **`DATA_ROOT` stays OUTSIDE the repo.** This still matters now that nothing
  is synced: no git operation, clone, or accidental `git add -f` can reach an
  image. **Do not move the data root back inside the repo.**
- **A git remote exists, and it is code-only.** Decided 2026-09-11 at Noah's
  request, deliberately before any real image existed. This supersedes the
  earlier "no git remote" rule for CODE. It relaxes nothing about images: no
  photo is ever committed, force-added, or pushed.
- **The ingest server is LAN-only and must be reachable on a PRIVATE network
  only.** It binds `0.0.0.0:8000`. No route serves image bytes, but `/health`
  returns the data-root path, `/recent` returns the capture log, and
  `/capture`, `/lock-angle`, `/unlock-angle` and `/regimen` are unauthenticated
  **writes** — anyone on the network can inject images into the dataset or flip
  the angle gate. The firewall rule is `-Profile Private` for that reason; see
  `docs/NETWORK_SETUP.md` §1.

  **Open gap as of 2026-09-12, and real photos now exist:** that rule is not
  installed, the Wi-Fi is classified *Public*, and what lets the phone through
  is a pair of Windows-auto-created "allow python.exe inbound" rules scoped to
  Public — so the server is reachable on *any* network the laptop joins. Fix
  before the next capture session, not "before day 1"; day 1 has happened.
- Backup goes to an external drive or local folder, never a git host and never
  a cloud service. `scripts\backup_data.ps1`.
- No image is ever sent to a third-party API. Not for labeling, not for
  inference, not for "just checking something".

## How Noah wants work run

- **Blunt over encouraging.** If a result is bad, say it is bad. If an idea is
  wrong, say so before he spends a week on it. No praise padding.
- State the verdict first, then the reasoning.
- **That is how REPORTS are written. Anything Noah reads AT THE RIG is the
  opposite: plain English, no jargon.** QA rejections, gate banners, upload
  errors. He is on a phone at 7am with the lamp still set up, and a message
  he has to decode gets overridden instead of acted on - which is how a bad
  frame enters the series. Structure it `what` / `fix` / `detail`: what is
  wrong in words, the physical action to take now, then the measurement -
  kept, but small and behind a toggle. Never drop the numbers: the QA
  thresholds are provisional and the measurement is how a wrong one gets
  caught. See `Problem` in `src/server/qa.py`; `tests/test_qa_messages.py`
  enforces it with a jargon blocklist.
- **Push back on a spec you think is wrong, before implementing it.** This has
  already paid for itself three times: center-membership does not partition
  overlapping tiles, float round-trips cannot be bit-exact through a
  translation, and `_spatial_subsample`'s real bug was a collapsing per-cell
  quota rather than the truncation it was reported as. A wrong fix that looks
  reviewed is worse than a known-wrong original.
- Build in reviewable chunks and verify each one before moving on.
- When a gate fails, name what failed and the cheapest next lever. Do not
  soften it.
- Flag the physical-world steps he has to do himself rather than working
  around them — rigging, printing, shooting. You cannot do those.
- Record settled decisions in `docs/DECISIONS.md` as they happen. A decision
  that lives only in a chat is lost to the next session.

## Open questions

- Face Mesh reliability at 60 deg (P0 decides; fallback 50 deg).
- Which registration method wins — decided by *simplest that clears the pooled
  gate*, not by lowest number.
- Main lens @ 30cm vs 2x @ 55cm (P0 lens test).
- Does iOS Safari alter uploaded files (`verify_upload_fidelity.py`).
- px/mm at video-stream resolution (`/probe`) — decides whether P1 keeps the
  ghost overlay or switches to native capture.
- **A pair no method can score currently sinks the verdict.** Gate eligibility
  requires scoring every pair, so one under-marked image makes every method
  ineligible and the verdict reads NO-GO — a data problem presented as a
  registration failure. Detect pairs scored by zero methods, drop them from the
  pool, and say so loudly above the verdict.
- QA thresholds in `config.py` are provisional and need retuning on real
  photos.
- ACNE04 dataset availability and license (P4, check before relying on it).
- The six region polygons are named but not yet defined in canonical
  coordinates (P3).
