# Build plan — Lesion Atlas

Markdown of `Lesion-Atlas-build-plan.pdf` (v3, 11 Sept 2026), which stays in
the repo as the formatted original. This version exists so the reasoning is
greppable and linkable. Where they differ, the PDF is the source of truth.

Geometry lives in [capture-geometry.md](capture-geometry.md).
Task order lives in [runbook.md](runbook.md).

---

## What this is

A daily skin-capture pipeline that finds acne lesions across six facial
regions, holds each lesion's identity across days in a fixed anatomical
coordinate frame, and turns months of that into a picture of how the skin is
actually changing.

Capture: phone browser → local server. Poses: frontal + L/R 60°. Regions: 6.
Zoom: none. Labels: 300–600 images. Fallback: laptop webcam.

---

## Where this fails

| Risk | Sev | Why it kills the project | Mitigation |
|---|---|---|---|
| **You stop shooting** | High | Everything downstream needs 60+ consecutive days. A three-week gap is unrecoverable. The most likely failure mode by a wide margin. | Capture under 60 s start to finish. Ghost overlay, no typing, no menus. Optimise this before anything else. |
| **Lighting drift** | High | Invisible until month three, when the chart shows a step change on the day you replaced a lightbulb. Redness is the signal; lighting is a confounder that fully swamps it. | Printed colour card in every frame, one fixed lamp, same time of day. Normalise per-frame against the card. |
| **Registration drifts across days** | Med | Without a reliable mapping into a fixed anatomical frame there is no lesion identity, and half of what you want is gone. | Phase 0 spike before anything else. Landmarks for coarse alignment, feature matching to a per-pose atlas for the fine fit, hard rejection of bad registrations, daily cross-pose check. |
| **Random train/test split** | Med | Consecutive days of the same face are near-duplicates. A random split puts Tuesday in train and Wednesday in test; metrics look excellent and mean nothing. | Split by date. Hold out contiguous blocks of days, never individual images. |
| **Inconsistent labelling** | Med | If you cannot reproduce your own labels, the model's ceiling is your noise floor — and you will misread label noise as model error for weeks. | Write the rubric before labelling. Measure self-agreement at image 100. Under ~0.75, fix the rubric, not the model. |
| **Scale drift** | Low | Standing 5 cm closer makes everything 15% bigger. Size thresholds in pixels become meaningless and match gates misfire. | The fiducial that fixes lighting fixes this too. Work in millimetres, never pixels. |

---

## Sequencing

The detector is in v1, but it cannot be built first — it needs a few hundred
labelled photos of your face that do not exist yet. So capture starts
immediately (every day you delay is a day permanently missing from your
longitudinal data), labelling runs alongside it, and the detector lands around
week 5.

Registration comes before all of it, because a perfect detector with broken
registration gives you daily counts and no lesion identity — which is half of
what you asked for.

---

## Phase 0 · Spike — go / no-go · week 1

Do not build the app yet. Prove the load-bearing things work, with throwaway
code and throwaway photos.

**What you shoot here is calibration, not a session.** One pose — left 60° —
five separate times across three days, setting up from scratch each time. One
pose, five repeats. Repeatability is the whole point: one shot tells you
nothing about whether the pose lands in the same place tomorrow.

Three questions:

1. **Does Face Mesh hold at 60°?** MediaPipe must fire on all five repeats
   with stable landmarks. If it drops out, step down to 50° — that constraint
   sets the angle, not aesthetics.

2. **Does fine registration land?** Freeze one of the five as the atlas and
   register the other four to it. Coarse-align with landmarks, then refine:
   ORB/SIFT + RANSAC → LightGlue with DISK → LoFTR (detector-free, the one
   that works on low-texture skin).

3. **Which lens?** Same pose on the main lens at ~30 cm and the 2× at ~55 cm.
   Compare sharpness at the sideburn and at the nose within the same frame —
   depth of field is what you are testing.

Measure honestly: mark 8–10 corresponding points by hand (moles, freckles,
scars — excellent anchors and your ground truth), apply the estimated
transform, report median and p95 reprojection error in millimetres.

Homography assumes a plane; a turned face very much is not one. Expect
structured residuals toward the frame edges and step up to a thin-plate spline
fit on the matched points.

> **GATE — median error under 1.5 mm, p95 under 3 mm.** Worse, and you cannot
> distinguish adjacent lesions; reduce yaw and retry before changing anything
> else.

---

## Phase 1 · Capture rig and app · week 1–2

Two halves, and the physical half matters more.

**The rig.** One diffused lamp, fixed position. Blinds shut, overheads off.
Same time of day — erythema tracks temperature, exercise and time since
showering, all of which track the clock. Printed ArUco marker plus a small
colour patch on matte paper, taped to a headband or stand in frame: one piece
of cardstock buys absolute mm scale, white-balance normalisation and a
registration anchor. Highest leverage per dollar in the project. Floor tape
for feet, wall marks to turn toward.

**The app.** FastAPI serving a static page. Self-signed cert — `getUserMedia`
needs a secure context and `localhost` does not help you on the phone.
Three-step wizard: frontal → left 60° → right 60°, one tap each, auto-advance,
then a covariate row. No text entry.

**Ghost overlay.** Composite the previous session's frame for that pose over
the live camera at ~35% opacity and align to it by eye — including the yaw
angle, which is otherwise the hardest thing to reproduce. This is the
highest-value feature in the app: it moves the registration problem to capture
time, where a human solves it for free.

Tap-to-focus defaulted to the region the pose owns. Request manual exposure,
fixed white balance and torch via `applyConstraints`, and **log what was
actually granted** — Android Chrome honours some, iOS Safari essentially none.

> **GATE — a full session under 60 seconds, done seven days running.**

---

## Phase 2 · Labelling · week 2–6 · the bulk of the work

Runs in parallel with daily capture. Budget 15–30 hours. The most common place
this kind of thing stalls.

**Write the rubric first.** "What counts as a pimple" is a documentation
problem, not an algorithm problem. Minimum diameter in mm (something like
1.5). Must be elevated or erythematous — flat and brown is a mark, not an
active lesion. Explicit negatives: moles, pores, ingrown hairs, razor bumps,
scars, shaving irritation. A named negative class teaches far more than an
unlabelled region.

**Classes:** `papule`, `pustule`, `comedone`, `mark`, `other`.
You can merge classes later. You cannot split them later without relabelling.

**The tool.** Click to place, scroll to size the radius, number keys for
class. Store centre + radius in image pixels; convert to boxes at train time.

**Active learning.** Label ~100 by hand, train a weak model, have it pre-label
the next batch and only correct it. Three to five times faster per image, and
it is what makes 500 tractable.

> **GATE — at image 100, re-label 20 images from two weeks earlier without
> looking at the old labels. Agreement (matched within 2 mm) under ~0.75 means
> the rubric is broken and no model will beat it.**

---

## Phase 3 · Canonical space · week 3–4

Productionise the spike. Each pose gets a fixed coordinate frame in
millimetres that every session of that pose maps into.

Three atlases, one per pose. Freeze a high-quality session as each and never
change them — changing an atlas invalidates every historical coordinate
under it.

Per image: Face Mesh landmarks for coarse alignment → detect fiducial,
normalise colour and scale → feature-match to the pose atlas → similarity,
then homography, then TPS if residuals warrant → store the transform and its
residual stats.

**Reject, do not average.** Residual over threshold means flag the session
`unregistered` and re-shoot. A silently bad registration corrupts lesion
identity in ways you will not notice for months.

Define the six region boundaries once in canonical coordinates, each tagged
with its owning pose.

**Daily cross-pose check.** A lesion in the mid-cheek band appears in both the
frontal and the turned pose. Map both into a shared head frame and measure the
disagreement. That number is your honest registration error — and unlike the
phase 0 spike, it reports itself every single day forever.

Sanity check: warp three different days into canonical space and blink between
them. Moles should not move. If they do, stop and fix it.

---

## Phase 4 · Detector · week 5–7

Small-object detection on a low-data, single-subject problem.

**Tile everything.** Lesions occupy a tiny fraction of the frame. Cut into
640 px tiles at ~20% overlap, train on tiles, run sliced (SAHI-style)
inference and merge with NMS. This is the difference between a model that
works and one that does not — and it is also why you do not need optical zoom:
the detector never sees a full-face image anyway.

Model: YOLOv8n/v11n or RT-DETR. Small model, small dataset; resist going
bigger.

Pretraining: look for ACNE04 (Wu et al., ~1,450 images with lesion boxes).
**Verify it is still publicly available and check the licence.** If you get
it, pretrain there and fine-tune on your own face; if not, COCO weights and
heavier augmentation.

Augmentation: aggressive colour jitter and lighting simulation, mild rotation,
flips. Avoid heavy geometric warps — they distort lesion morphology, which is
your signal.

Train **one** model across all poses with pose as an input feature. Three
per-pose models would each get a third of the data.

**Split by date.** Contiguous held-out blocks of days. Say it out loud before
writing the split function.

**Metrics:** P/R at 2 mm centre distance (IoU-based mAP is a poor fit for
near-circular lesions); count MAE per region per session; day-to-day Δ error.
Tune the confidence threshold toward precision — a false positive creates a
phantom lesion that enters tracking and pollutes your history; a false
negative is a gap that tracking bridges anyway.

> **GATE — precision above ~0.8 at recall above ~0.7, on held-out days.**

---

## Phase 5 · Lesion identity across days · week 7–8

What makes this a tracker rather than a counter. In canonical space it is
nearly easy — which is exactly why phase 0 came first.

Project every owned detection into canonical mm. Match today's detections
against **active lesions' last known positions** — not against yesterday's
frame — with the Hungarian algorithm. Matching to lesion state rather than to
the previous frame is what makes missed days harmless.

Gate at ~4 mm. Pimples do not move; a match beyond that is wrong.

Lifecycle: born once an unmatched detection recurs in 2 of 3 consecutive
sessions (single-session detections are overwhelmingly false positives).
Resolved after 3 consecutive sessions absent, backdated to the last sighting.

Record class per observation and you get lifecycle traces — papule → pustule
→ mark → gone — with durations. This is the genuinely interesting data, and no
consumer app gives it to you.

**Log covariates from day one.** Sleep, stress, new products, diet flags,
whether you worked out, whether you skipped washing. Ten seconds in the
capture app. You cannot collect these retroactively, and in six months they
are the difference between "here's a chart of my acne" and "here's what moves
my acne".

---

## Phase 6 · Visualisation · week 8+

In descending order of how much you will actually look at them:

1. **Active count over time**, stacked by region, 7-day rolling mean overlaid.
   Daily counts are noisy and the raw line will mislead you.
2. **Canonical density heatmap** — kernel density of lesion-days per region,
   so bright means "lesions recur here", not "one was here once". Small
   multiples by month makes the distribution shift directly visible.
3. **Lesion swimlanes** — one horizontal bar per lesion from birth to
   resolution, coloured by class over time. The most information-dense view.
4. **Duration distribution.** "Are they resolving faster?" is a sharper
   question than "are there fewer?" and it answers sooner.

Streamlit or a static Plotly page. Do not build a web app; you are the only
user.

---

## Phase 7 · Later, where the ML gets interesting

Ordered by value, not by how cool they sound.

- **Self-supervised pretraining.** You will have thousands of unlabelled skin
  images. Fine-tune DINOv2 on them and use the features as a detector
  backbone. Highest-value extension, and it uses the asset you will have most
  of.
- **Resolution forecasting.** Given a lesion's first two days, predict whether
  it resolves within five. A clean supervised problem your own tracking data
  generates for free.
- **Covariate analysis.** Lagged regression of lesion births against logged
  covariates. Not causal, but n=1 with dense daily sampling over months is a
  better dataset than most acne studies have.
- **Severity regression** instead of discrete classes.

**On generative AI.** It has essentially nothing to offer the core problem.
Two legitimate uses: a VLM as a pre-labelling assistant in phase 2 (real,
useful), and a natural-language query layer over your own database (fun, zero
analytical value). Diffusion-based synthetic lesion generation is what people
reach for — on a single-subject dataset it will almost certainly hurt, because
the model learns your generator's artefacts rather than your skin.

---

## Schema

```
sessions      id, captured_at, device, lamp_ok, covariates_json
images        id, session_id, pose, path, exif_json, device, lens,
              qa_score, reg_status, reg_median_err_mm, transform_json
detections    id, image_id, px_x, px_y, px_r, cls, score, model_version,
              canon_x_mm, canon_y_mm, canon_r_mm, region, owned
lesions       id, region, first_session_id, last_session_id,
              status, canon_x_mm, canon_y_mm
observations  id, lesion_id, detection_id, session_id, cls, severity
labels        id, image_id, px_x, px_y, px_r, cls, labeler, created_at
```

Detections are per-image and disposable — run a better model, regenerate them.
**Lesions are the durable identity.** Observations is the join, and it is what
every chart reads from.

`detections.owned = false` means the lesion was visible in this pose but
another pose owns that region. Kept for the daily cross-pose check, excluded
from every count. This one flag is what stops you double-counting everything
in the mid-cheek overlap.

---

*Nothing here is a diagnostic tool. A daily count that jumps around is
measurement noise well before it is your skin changing, which is why every
view in phase 6 shows a smoothed trend rather than a single day.*
