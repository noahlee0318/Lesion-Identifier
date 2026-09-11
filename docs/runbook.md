# Runbook — every task, in order

Markdown of `Atlas-Runbook.pdf` (11 Sept 2026), which stays in the repo as the
formatted original. The build plan says *why*; this says *do this, then this*.

Where they differ, the PDF is the source of truth.

---

## Are you collecting training data during calibration?

Not from the calibration shots themselves — but you start collecting real data
two or three days in, well before the app exists.

The five phase-0 shots are throwaway. One pose, one side, taken before the rig
is locked and before `RUBRIC.md` exists. They answer "does 60° work" and
nothing else.

But calendar time is the one input you cannot buy back later. Sixty days of
history takes sixty days, and no amount of effort in week 6 recovers a session
you did not shoot in week 1. So the order is:

- **Days 1–2** — rig up, shoot the five calibration repeats, run Face Mesh,
  lock the angle. The shooting is fast; the three days are the point.
- **The day after the angle locks** — start shooting real three-pose sessions
  manually, with your normal camera app, into `data/raw/YYYY-MM-DD/`. These
  are real data: day 1 of your longitudinal series and the first of your
  training images.
- **Meanwhile** — the heavier registration spike and then the capture app get
  built while sessions accumulate in the background.

> **Corrected 2026-09-11.** This section originally read "Days 1–2 … Day 3
> onward", which contradicts the build plan's "five separate times across
> three days" and left the start of daily capture ambiguous. **The gate is
> the angle lock, not a calendar day.** Five repeats need at least three days
> to spread across, so the lock lands on day 3 at the earliest and daily
> capture starts the day after. `PHASE0_CHECKLIST.md` has the table.

> ### The one ordering rule that matters
>
> **Do not start real sessions until the angle is locked.** If phase 0 pushes
> you from 60° to 50°, every session shot at 60° beforehand is at the wrong
> angle and cannot join the series. That is the only reason to wait — and it
> is a three-day wait, not a two-week one. Everything else about the app is
> automation of something you can already do by hand.

Expect roughly 5–8 manually-shot sessions before the capture app takes over.
Name the folders correctly from the first one and they flow into the pipeline
untouched.

---

## Timeline · 8 weeks at 8–12 hrs

| Phase | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 |
|---|---|---|---|---|---|---|---|---|
| 0 Spike & angle lock | ██ | | | | | | | |
| **— Daily capture (manual)** | ██ | ██ | | | | | | |
| 1 Capture rig & app | ██ | ██ | | | | | | |
| **— Daily capture (app)** | | ██ | ██ | ██ | ██ | ██ | ██ | ██ |
| 2 Labelling | | ██ | ██ | ██ | ██ | ██ | | |
| 3 Canonical space | | | ██ | ██ | | | | |
| 4 Detector | | | | | ██ | ██ | ██ | |
| 5 Lesion tracking | | | | | | | ██ | ██ |
| 6 Visualisation | | | | | | | | ██ |

The daily-capture rows never stop once started. Everything else is build work
that happens around them. They are separate rows because that is the point:
**capture is not a phase you finish.**

---

## Phase 0 · Spike & angle lock · week 1

- [ ] Print an ArUco marker (4×4_50, ~30 mm) and a small colour patch on
      **matte** paper — gloss will specular and ruin the white balance reference
- [ ] Mount both on a headband or small stand that sits in frame for every pose
- [ ] One diffused lamp in a fixed position; blinds shut, overhead lights off
- [ ] Floor tape for your feet; wall marks left and right at 60° to turn toward
- [ ] Write down the setup: camera distance, lamp position and height, time of
      day. You will need to rebuild this exactly after any disruption
- [ ] Install: python, opencv, mediapipe, numpy, torch with CUDA, kornia
- [ ] Shoot one pose — **left 60° — five separate times over three days**,
      setting up from scratch each time, on both the main and 2× lens
- [ ] Run MediaPipe Face Mesh on all five. Does it fire every time? Are the
      landmarks stable across repeats?
- [ ] **GATE — if Face Mesh drops out, step down to 50° and re-shoot.
      LOCK THE ANGLE HERE.**
- [ ] *Runs in parallel:* start shooting real 3-pose sessions manually into
      `data/raw/YYYY-MM-DD/`. Every day from now on, forever
- [ ] Hand-mark 8–10 corresponding points across the five calibration images
      (moles, freckles, scars) as registration ground truth
- [ ] Freeze one of the five as the atlas; register the other four to it —
      ORB/SIFT + RANSAC first, then LightGlue + DISK, then LoFTR
- [ ] Compute median and p95 reprojection error in millimetres for each method
- [ ] If homography residuals are structured (worse toward the edges), fit a
      thin-plate spline instead
- [ ] Compare lens sharpness at the sideburn vs the nose within one frame.
      Pick a lens and a working distance
- [ ] **GATE — median error < 1.5 mm and p95 < 3 mm, or reduce yaw and retry**
- [ ] Write `RUBRIC.md`: minimum lesion size in mm, the
      elevated-or-erythematous rule, and explicit negatives, each with an
      example crop

---

## Phase 1 · Capture rig & app · week 1–2

- [ ] Install `mkcert`, generate a local cert, trust it on your phone —
      `getUserMedia` needs a secure context and `localhost` will not help you
- [ ] FastAPI skeleton serving a static page over HTTPS on the LAN
- [ ] SQLite init: `sessions` and `images` tables
- [ ] `POST /capture` — writes the image plus a sidecar JSON with device,
      lens, granted constraints, timestamp
- [ ] Phone page: `getUserMedia` on the rear camera, full-resolution stills
- [ ] Three-step wizard: frontal → left 60° → right 60°, one tap each,
      auto-advance
- [ ] Ghost overlay — fetch the previous session's frame for the current pose,
      composite at ~35% opacity over the live view
- [ ] Tap-to-focus, defaulted to the region that pose owns
- [ ] `applyConstraints` for manual exposure, fixed white balance, torch —
      and log what was actually granted, because iOS will grant almost none
- [ ] Covariate row after the third shot
- [ ] Automatic QA on upload: blur, exposure clipping, fiducial found —
      reject and re-prompt on the spot, not three weeks later
- [ ] Webcam fallback script writing to the same paths, tagged `device` so
      those sessions are never silently mixed in
- [ ] Migrate your manually-shot sessions into the database
- [ ] **GATE — a full session in under 60 seconds, done seven days running**

---

## Phase 2 · Labelling · week 2–6 · the bulk of the work

- [ ] Finish `RUBRIC.md` with real example crops from your own photos,
      including the borderline cases
- [ ] Build the labelling tool: canvas view, click to place a marker, scroll
      to size its radius, number keys 1–5 for class, undo, next/prev
- [ ] `labels` table storing centre and radius in image pixels
- [ ] Label the first ~100 images entirely by hand
- [ ] **GATE — re-label 20 images from two weeks earlier without looking at
      the originals. Agreement below ~0.75 means fix the rubric, not the model**
- [ ] Train a weak model on those 100 and wire up pre-labelling
- [ ] Add correct-mode to the tool: accept, nudge, delete, add. This is where
      the 3–5× speedup comes from
- [ ] Label to 300–600 images total, in batches, retraining as you go

---

## Phase 3 · Canonical space · week 3–4

- [ ] `calib.py` — ArUco detection, px/mm scale, colour normalisation
- [ ] Pick and freeze three atlas images, one per pose. Never change them
- [ ] `register.py` — landmark coarse alignment → feature match to the pose
      atlas → similarity, then homography, then TPS if residuals warrant
- [ ] Reject-don't-average: residual over threshold sets
      `reg_status = unregistered` and prompts a re-shoot
- [ ] `regions.py` — the six region polygons in canonical coordinates, each
      tagged with its owning pose
- [ ] Daily cross-pose check: map mid-cheek detections from both poses into a
      shared head frame, log the disagreement
- [ ] Blink test — warp three different days into canonical space and flip
      between them. Moles must not move

---

## Phase 4 · Detector · week 5–7

- [ ] Check whether ACNE04 is still publicly available, and read its licence
- [ ] Tiling: 640 px tiles at ~20% overlap, with label remapping
- [ ] Date-based split function. Contiguous held-out blocks of days — write
      this **before** the training script, not after
- [ ] Train YOLOv8n/v11n or RT-DETR — one model across all poses, with pose as
      an input feature
- [ ] Augmentation: heavy colour jitter and lighting simulation, mild
      rotation, flips. No aggressive geometric warps
- [ ] Sliced inference across tiles with NMS merge at the seams
- [ ] Metrics: precision/recall at 2 mm centre distance, count MAE per region,
      day-to-day change error
- [ ] Tune the confidence threshold toward precision
- [ ] **GATE — precision above ~0.8 at recall above ~0.7, on held-out days**

---

## Phase 5 · Lesion tracking · week 7–8

- [ ] Project every owned detection into canonical mm coordinates
- [ ] Hungarian matching against active lesions' last known positions
- [ ] Gate matches at ~4 mm
- [ ] Lifecycle state machine: born on recurrence in 2 of 3 consecutive
      sessions; resolved after 3 consecutive absences, backdated
- [ ] `lesions` and `observations` tables, with class per observation
- [ ] Manual audit — pick 10 tracked lesions and verify by eye across their
      whole span. Tracking bugs are invisible in aggregate and obvious in
      individual traces

---

## Phase 6 · Visualisation · week 8+

- [ ] Active count over time, stacked by region, 7-day rolling mean
- [ ] Canonical density heatmap of lesion-days, small multiples by month
- [ ] Lesion swimlanes — one bar per lesion, birth to resolution
- [ ] Duration histogram
- [ ] Wrap in Streamlit or a static Plotly page

---

## The three interfaces

### 1 · Daily capture (phase 1)

The **ghost is the whole design**. Everything else is chrome around it. You
match your face to a translucent copy of yesterday's frame, which is why the
yaw angle stays reproducible without a jig.

- **Live alignment readout** — run Face Mesh on the preview stream, compare
  landmark positions to the ghost's, show the residual in mm. Turns green
  under ~2 mm. The single feature that will most improve registration quality,
  because it fixes the problem before the shutter fires instead of rejecting
  the shot afterward.
- **Three dashes, not a step counter.** The session is short enough that you
  should never have to read a number to know where you are.
- **No text entry anywhere.** Covariates are taps with three options each.
  The moment a keyboard opens, the 60-second budget is gone.
- **Reject on the spot.** If blur or exposure QA fails, re-prompt that pose
  immediately. A bad frame you discover in week 6 is a lost day.

**Build order:** wizard and upload first, ghost overlay second, live alignment
readout last.

### 2 · Calibration & registration review (phases 0 and 3)

One screen, two jobs. In phase 0 you drive it manually to compare methods and
lock the angle. In phase 3 it becomes the review queue for sessions that
failed automatic QA — same view, same numbers.

Readout row: median err, p95 err, matches, transform, scale (px/mm), Face Mesh
status.

- **Everything in millimetres.** Pixels are meaningless across sessions once
  your standing distance varies by a couple of centimetres.
- **Blink compare is the honest test.** Numbers can look fine while the warp
  is subtly wrong. Flipping between atlas and warped image at ~2 Hz makes any
  real misregistration jump out immediately — moles will visibly jitter.
- **"Flag for re-shoot"** has to be one click and available the same day,
  while re-shooting is still possible.

### 3 · Labelling (phase 2)

- **Optimise for keystrokes, not looks.** You will place several thousand
  markers. Every extra click multiplies by 3,000.
- **Label one tile at a time**, not the whole frame. The detector trains on
  640 px tiles anyway, and labelling at the zoom the model sees is faster and
  more consistent.
- **Show the radius in millimetres** while sizing.
- **Pre-labels are visually distinct** (dashed until confirmed), so accepting
  is deliberate. Once the model is decent you will be tempted to accept-all
  and skim — that is how a model starts training on its own mistakes.
- **Keep the rubric one key away.**
- **Track time per batch.** If it is not falling once pre-labelling is on, the
  pre-labeller is not good enough to be worth the correction overhead.

---

*The one thing on this page that is genuinely urgent is starting daily
capture. Everything else can slip a week without consequence; a missed session
is gone permanently.*
