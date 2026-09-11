# Lesion Atlas

A personal longitudinal acne-tracking system. Daily 3-pose facial capture →
detect acne lesions across 6 face regions → hold lesion identity across days
in a fixed anatomical coordinate frame → visualise count and spatial
distribution change over months.

Single subject. Runs entirely on one Windows 11 laptop with an NVIDIA GPU.
Capture device is an iPhone 15. **No image ever leaves the machine.**

> **Status: phase 0 (spike). Tooling is built and tested against synthetic
> fixtures. Waiting on real calibration photos.**

---

## What makes this different from a photo diary

Three things, in order of difficulty:

1. **Millimetres, not pixels.** A printed 30 mm ArUco marker sits in every
   frame, so every measurement converts through that image's own scale.
   Standing 5 cm closer makes everything 15% bigger; without the marker,
   size thresholds are meaningless across days.

2. **A fixed anatomical frame.** Each pose gets a frozen atlas image. Every
   session registers into it, so a lesion has coordinates that mean the same
   thing in March and in June. This is what turns counting into *tracking*.

3. **Lesion identity across days.** Hungarian matching in canonical
   millimetres against active lesions' last known positions — not against
   yesterday's frame — so a missed day is harmless. You get lifecycle traces:
   papule → pustule → mark → gone, with durations.

---

## Phase 0 answers three questions

| # | Question | Gate | Tool |
|---|---|---|---|
| a | Does MediaPipe Face Mesh fire reliably with stable landmarks at 60° yaw? | 5/5 detections, cheek/temple anchors stable | `src/facemesh_check.py` |
| b | Can two photos of the same face on different days be registered accurately enough? | **median < 1.5 mm, p95 < 3 mm** | `src/register.py` |
| c | Main lens @ ~30 cm or 2× @ ~55 cm? | more detail at the sideburn without losing the nose | `src/lens_compare.py` |

Plus two iOS questions that decide phase 1's architecture — does Safari alter
uploaded files (`scripts/verify_upload_fidelity.py`), and what resolution does
`getUserMedia` actually grant (`/probe`).

If (a) fails, the angle steps down to 50° and everything downstream shifts.
That is why nothing else gets built first.

---

## Setup

**Python 3.12 is required** — mediapipe publishes no 3.13/3.14 wheels.

```powershell
# from the repo root
<python3.12> -m venv .venv
.\scripts\install_deps.ps1      # torch from the CUDA index, then the rest
python scripts\fetch_weights.py # caches every model in-repo; run once online
python scripts\verify_env.py    # blunt PASS/FAIL per requirement
```

`verify_env.py` fails loudly if CUDA is missing rather than silently falling
back to CPU.

### Where the data lives

`C:\LesionAtlas\data` — **outside this repo, on purpose.** The repo folder is
inside OneDrive, and photos of your face must not sync to a cloud service.
Override with the `LESION_ATLAS_DATA` environment variable.

```
<DATA_ROOT>/
  raw/
    calib/              phase 0 calibration shots
    YYYY-MM-DD/         daily sessions + sidecar JSON per image
  derived/controlpoints/
  atlas/                frozen reference frame per pose — never edit
  atlas.sqlite3
```

---

## Daily use

```powershell
python -m src.server.main
```

Prints the LAN URL and a QR code. Open it on the phone, attach the photos
from your camera roll, answer the tap rows, upload. Per-image QA comes back
immediately so a bad frame gets reshot in the same sitting.

To make it always-on (run once, as administrator):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_autostart.ps1 -WithFirewall
```

Back up the data — the code is a weekend, the images cannot be recreated:

```powershell
.\scripts\backup_data.ps1 -Destination E:\LesionAtlasBackup -Install
```

---

## Phase 0 commands

```powershell
# printable calibration target (PDF + PNG + sidecar JSON)
python -m src.fiducial generate
python -m src.fiducial verify photo_of_printed_sheet.jpg

# (a) does Face Mesh hold at 60 deg?
python -m src.facemesh_check <DATA_ROOT>\raw\calib

# hand-mark control points — held-out ground truth
python -m src.controlpoints <DATA_ROOT>\raw\calib

# (b) the registration bake-off + GO/NO-GO
python -m src.register bakeoff --ref REF.jpg --targets <DATA_ROOT>\raw\calib
python -m src.register review  --ref REF.jpg --target T.jpg --method tps

# (c) lens comparison
python -m src.lens_compare --main SHOT_main.jpg --tele SHOT_2x.jpg

# does iOS Safari alter uploads?
python scripts\verify_upload_fidelity.py --usb IMG.JPG --uploaded STORED.jpg
```

Reports land in `reports/` with the verdict at the top of each.

---

## Layout

```
src/
  config.py           paths, locked constants, the data root
  fiducial.py         printable target, ArUco detection, px/mm
  facemesh_check.py   question (a) — landmarks, pose, occlusion, stability
  controlpoints.py    click tool for ground-truth moles
  register.py         question (b) — 6-method bake-off, TPS, blink compare
  lens_compare.py     question (c) — cycles/mm per region per lens
  server/
    main.py           FastAPI ingest, plain HTTP, LAN only
    db.py             sessions / images / regimen_events + calibration counter
    qa.py             blur, clipping, fiducial — per upload
    static/           one page, no framework, no build step
scripts/              install, weights, autostart, backup, fidelity check
docs/                 build plan, runbook, geometry, checklists
tests/                synthetic fixtures with known ground truth
```

---

## Rules that are not up for debate

Full list in [CLAUDE.md](CLAUDE.md). The ones that bite hardest:

- **Millimetres, never pixels**, in anything a human reads. Convert via the
  per-image ArUco scale, never a global constant.
- **Never re-encode an image.** The server stores received bytes unmodified —
  no resize, no EXIF strip. SHA-256 per file.
- **Control points are held-out ground truth.** Never fit a transform using
  them.
- **Reject, don't average.** A bad registration is discarded, never blended
  into an atlas.
- **Split by date, never randomly.** Consecutive days are near-duplicates; a
  random split leaks badly and produces fake metrics.
- **Log covariates from day one.** They cannot be collected retroactively.
- **Adherence is the #1 risk.** Capture must stay under 60 s.

---

## Privacy

This repo contains **no images** and is configured so it cannot. `.gitignore`
blanket-bans every raster and video format with no exceptions, and the data
root lives outside the repo entirely.

The photos never go to a cloud service, never to a git host, and never to a
third-party API — not for labelling, not for inference, not for "just checking
something".

---

*Nothing here is a diagnostic tool. A daily count that jumps around is
measurement noise well before it is your skin changing, which is why every
view shows a smoothed trend rather than a single day.*
