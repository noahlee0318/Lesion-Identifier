# Personal ML workflow

This is an implemented structure with an executable ML baseline. It is not an
accurate acne product yet. Calibration alone cannot make a detector work for a
new person: each person needs reviewed annotations and an honest evaluation on
unseen days. Even passing the provisional gate is not clinical validation.

## What's implemented

1. **Local profile:** separate photo and annotation records per profile in
   browser IndexedDB. These are not cloud accounts or secure separation from
   another person using the same browser. Names are optional nicknames.
2. **Capture and scale:** save original JPEG/PNG bytes, manually date and tag the
   view, then tap the two ends of a measured marker edge. This records scale,
   not alignment quality or a passed legacy registration gate. Practice
   calibration photos are excluded from training.
3. **Annotation:** mark centers and disc diameters in millimetres. Export stores
   circles in oriented full-image coordinates, never preview coordinates. A
   photo must be explicitly reviewed before training; a reviewed empty label
   list is an intentional negative example. Uncertain images stay drafts.
4. **Dataset export:** one JSON file containing original image bytes as base64,
   SHA-256 hashes, profile identity, dates, view, scale and labels. This file is
   private: save outside Git and cloud-synced folders. Export saved work before
   clearing browser storage. Restore verifies hashes and dimensions and creates
   a separate profile, rather than overwriting existing work. Restored profiles
   get new IDs; keep the original frozen export to reproduce an old model run.
5. **Real local training:** a small PyTorch convolutional network predicts a
   lesion mask. It learns from scratch on one profile and one view; no weights
   or images are downloaded. Pixel targets come from the reviewed discs.
6. **Evaluation:** validation days choose the mask threshold from 0.3, 0.5 and
   0.7. Test days are used only by the separate evaluation command. Model
   weights, dataset snapshot, splits and training settings have provenance.
7. **Experimental inference:** output candidate centers in millimetres for an
   individual calibrated photo from the same profile and view. These are not
   trusted spot totals and are not added to the home-page count cards.

The workspace blocks connection requests with its Content Security Policy.
Use public HTTPS or localhost; plain HTTP LAN access is unsupported for these
browser cryptography tools.

The public workspace uses no upload endpoint. Its JavaScript never sends photo
bytes, labels, scales, dates or profile names to the Worker or chatbot. Browser
storage is specific to the device, browser and origin: localhost and the public
domain have different datasets. A future domain change requires export/restore.
Each photo is limited to 20 MB and each profile to 200 MB of original bytes;
large JSON backups require substantial memory.
Restore currently accepts backups up to 300 MB. This is an early dataset tool,
not a long-term photo archive; keep local original-photo backups too.

## User steps

Open **Personal model workspace** from the home page. Create a profile, add a
photo, measure scale, label spots, and save as reviewed only after checking all
visible spots. Save drafts before switching photos or profiles. Negative images
are valuable, but do not use an unlabeled acne photo as a negative.

Keep the marker near the cheek's distance from the camera and use consistent
capture conditions. A single scale is only an approximation over a curved face.
Side views remain experimental; the legacy angle/registration research still
matters for later tracking. Manual scale calibration is not a substitute for it.

Collect reviewed session photos over time. The default split is chronological:
older training data, 3 excluded calendar days, 14 validation calendar days,
3 excluded days, 14 test calendar days. Roughly 35 calendar days are needed,
with actual reviewed photos in all three blocks. Sparse capture or missing
positive examples can require more time. Practice photos and unreviewed images
cannot satisfy readiness. Every view is checked independently.

## Local training commands

Use the existing Python 3.12 environment and dependencies. Save your export as
`C:\LesionAtlas\data\personal\dataset.json` (or another private local path).
From the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.personal_ml inspect C:\LesionAtlas\data\personal\dataset.json
.\.venv\Scripts\python.exe -m src.personal_ml train C:\LesionAtlas\data\personal\dataset.json C:\LesionAtlas\data\personal\runs\v1 --pose frontal --epochs 10
.\.venv\Scripts\python.exe -m src.personal_ml evaluate C:\LesionAtlas\data\personal\dataset.json C:\LesionAtlas\data\personal\runs\v1
.\.venv\Scripts\python.exe -m src.personal_ml predict C:\LesionAtlas\data\personal\dataset.json C:\LesionAtlas\data\personal\runs\v1 --image-id PHOTO_ID_FROM_EXPORT
```

Training defaults to CUDA and errors if it is unavailable. Choose `--device cpu`
explicitly for slower CPU training. It does not silently change devices.
Evaluation and prediction run on CPU. A training run refuses an existing output
directory and must live outside the repository. Evaluation refuses to overwrite
its report. Keep the exact exported dataset: edited labels or newly added photos
change its hash and cannot be substituted in an existing run's evaluation.

Run artifacts:

| Artifact | Contents |
|---|---|
| `weights.pt` | Learned CNN parameters; loaded with `weights_only=True` |
| `model.json` | Profile/view, dataset and weight hashes, split IDs/day counts, seed, loss history, threshold, validation metrics |
| `evaluation.json` | Test precision/recall/F1, TP/FP/FN, per-photo breakdown and provisional gate result |

Source files: `src/personal_ml.py` is the local ML pipeline;
`cloudflare/public/personal*` is the browser workspace. The new workflow is
independent of the legacy SQLite ingest dataset and never changes its records.

## What the baseline measures

The CNN has three convolutional layers and uses 256×256 RGB previews, Adam,
weighted binary cross-entropy (positive weight 10), and seed 17. Original image
bytes are untouched. It may lose small lesions during resizing and merge
adjacent lesions. This is a runnable starting point, not the production tiled
detector described in the original build plan. No augmentation or shared
pretrained model is included yet. A seed is recorded, but bit-exact GPU
reproducibility across hardware/runtime versions is not promised.

Thresholded connected components with at least two preview pixels produce
candidate centers. One-to-one matching against labeled centers uses a fixed
2 mm tolerance. Metrics pool TP/FP/FN over every test photo, including negative
photos. Undefined precision or recall is `null`, not a fabricated perfect score.
Counts are per photo; summing front and side views would double-count regions.
No anatomical region ownership or longitudinal identity matching is implemented
in this baseline.

The provisional engineering gate requires precision > 0.8 and recall > 0.7,
at least 20 labeled test lesions, and at least five test capture days. Otherwise
the result is `gate_failed` or `insufficient_evidence`. Those sample minimums
are a conservative guard, not a proof of adequate statistical power. A successful
run never automatically marks the website's model validated or enables counts.
Repeatedly comparing models on the same test set introduces selection bias;
use future untouched days before making a new performance claim.

## Still to build

- Better label rubric, zoom/edit tools, annotation agreement checks and quality QA.
- A tiled/pretrained detector that retains fine detail and improves data efficiency.
- Automated per-user geometric calibration and alignment validation.
- Browser-compatible model export/runtime and trustworthy model promotion UI.
- Registered cross-day tracking, region ownership, deduplication and actual daily counts.
- Representative real-user evaluation; neither this baseline nor a model trained
  on one user proves performance on another user.

Synthetic tests exercise real optimization, evaluation and prediction plus
profile isolation, date gaps, duplicate rejection, immutable outputs and changed
dataset/weight rejection. Synthetic success is only software verification.

## Verification for this implementation

The repository suite passed 58 checks: 10 Python unittest checks (including
four personal-ML tests), 21 legacy tiling checks, 19 legacy split checks, and
eight Node checks. The 256×256 CPU synthetic pass used 40 generated images with
empty, one-spot and two-spot cases. Two training epochs reduced loss from
0.8228 to 0.7972, but test detection was poor: TP 0, FP 14, FN 14, precision and
recall 0. The report correctly returned `insufficient_evidence` because there
were only 14 labeled test lesions. This verifies execution and honest failure
reporting; it is not an acne accuracy result. No personal photos were used.

Visual browser verification remains pending: automatic browser approval was
blocked by the earlier usage-limit failure. Dataset selection/annotation/restore
should receive an interactive browser check before relying on the workspace
for an irreplaceable annotation project. Save original photos and exports.
