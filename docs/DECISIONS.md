# Decisions log

Chronological record of decisions that are settled, and why. `docs/runbook.md`
and `docs/build-plan.md` are the 11 Sept morning snapshots converted from the
artifact PDFs; anything below **supersedes them** where they disagree.

Add to this file rather than editing it — a decision that got reversed should
show as a later entry saying so, not vanish.

---

## 2026-09-11 — Ingest starts in P0, not P1

The phone→laptop connection splits into two pieces with very different costs:

- **Upload page (P0)** — `<input type="file">` + `POST /capture`. Shoot in the
  native Camera app, open a bookmark, select three photos. **Needs no HTTPS**:
  a file input is not a privileged API, so plain `http://<laptop-ip>:8000`
  works in Safari.
- **Capture app (P1)** — getUserMedia, ghost overlay, alignment readout. The
  entire mkcert / install-a-CA-on-the-phone ordeal exists only to satisfy
  `getUserMedia`, so skipping it in P0 is free.

Rationale for pulling it forward: the server, schema, date-directory writer,
sidecar JSON and upload QA all get shaken out on real photos during weeks 1–2,
so P1 adds only the hard part on top of tested foundations.

## 2026-09-11 — Server autostart is not optional

"Go start the server first" is the daily friction that kills adherence, which
is the project's #1 risk. Therefore: Task Scheduler task launching uvicorn at
logon, hidden, venv python, restart on failure, installed via a one-time
`scripts/install_autostart.ps1` run as admin. Rotating log at `logs/server.log`
so a 7am failure is diagnosable at 7pm. `GET /health` plus an explicit
"server unreachable" state on the page — silent failure is worse than loud.

Laptop asleep at shoot time is fine: photos stay on the phone and upload later.

## 2026-09-11 — Covariates are write-once; define before day 3

Changing a scale mid-series breaks comparability across the whole dataset.
Six daily 3-option tap rows: sleep (`<6h`/`6–8h`/`>8h`), stress
(low/normal/high), diet (clean/normal/junk-heavy), workouts
(none/light/hard-sweat), skipped wash (no/one/both), routine adherence
(followed/partial/skipped).

**Products is not a tap row.** Regimen change is a step function and will drive
most of the variance in the series — "started tretinoin Oct 4" cannot be
represented as low/normal/high. Separate dated `regimen_events` table
(added / stopped / dose changed).

## 2026-09-11 — `ImageCapture.takePhoto()` is unbuildable on the target device

Capture device is an iPhone 15 / iOS Safari. `ImageCapture` is Chromium-only.
The original P1 checklist item is not implementable as written.

Two fallbacks, in conflict:
1. Canvas grab off the `<video>` stream — keeps the ghost overlay, capped at
   video resolution, may be too coarse for ≥1.5mm lesions.
2. `<input capture="environment">` — full still resolution, loses the overlay.

**Decided by measurement, not argument**: `/probe` reports the resolution iOS
actually grants and measures px/mm at stream resolution against the ArUco sheet
at 30cm. ≥10 px across a 1.5mm lesion → option 1. Below → option 2.

Also: iOS Safari refuses manual exposure/WB constraints. Call `applyConstraints`
anyway and log what was granted vs refused.

## 2026-09-11 — Labels are stored in IMAGE coordinates, never tile coordinates

Tiles are a view, generated on demand, never a storage format. Per-tile storage
would mean changing `tile` or `overlap` later invalidates every label ever made
— months of annotation destroyed by a hyperparameter change.

## 2026-09-11 — Tile membership needs two functions, not one

*Corrects an earlier spec error.* Center-membership does NOT partition labels
when tiles overlap: a center in the 128px overlap band is inside two tiles,
four at a corner. So:

- `labels_in_tile` — every label centered in this tile, may appear in several.
  **Training.** A crop that renders a lesion must carry its label or it is a
  labelled false negative.
- `assign_labels_to_tiles` — exactly one owner, nearest tile center, ties by
  (row, col). **Counting and coverage.**

Same shape as `detections.owned` for the pose overlap band: kept everywhere
visible, counted in exactly one place.

## 2026-09-11 — Edge tiles are clamped, never zero-padded

Every frame would pad on the same two edges, so the black border correlates
with position in the frame — exactly the spurious signal a single-subject
dataset latches onto. The detector must inherit this: sliced inference on the
same clamped grid, NMS across the enlarged seams.

## 2026-09-11 — Split washout is measured in calendar days

Not in date-index. If a week is missed, the dates either side of the hole are
already far apart and no washout is needed; if capture is dense, three sessions
is three days. Only the calendar answers "are these two images near-duplicates".

Blocks are contiguous and chronological: train | gap | val | gap | test. No RNG
anywhere in the module; a source-level test asserts it.

## 2026-09-11 — Float round-trip through tile coordinates is not bit-exact

*Corrects an earlier spec error.* `(p - x0) + x0 == p` is unachievable in
IEEE-754 for arbitrary doubles — subtracting a multi-thousand offset drops
mantissa bits that adding it back cannot restore.

The guarantee is split: **integers bit-exact** (that is the persisted form, and
it is tested as exact), floats to < 1e-9 px. At 26 px/mm that is 4e-11 mm
against a 1.5 mm minimum lesion.

## 2026-09-11 — The phase 0 gate reads POOLED residuals, and recommends the simplest method that clears

Two compounding forms of selection-on-the-test-set were in the bake-off:
`write_report` sorted six methods by median and reported the minimum, and
method 5 picked its base matcher by control-point median and was then scored
on those same control points. With ~10 control points per pair, the minimum of
six is meaningfully below the true out-of-sample error, and this is the number
that decides whether the project proceeds.

Settled:

- **Pool per method across all pairs.** The gate statistic is the median/p90/
  p95/max of one concatenated residual vector. Not a median of per-pair
  medians — that is a weaker statistic that hides a single catastrophic pair,
  which is the failure the gate exists to catch. Per-pair numbers stay in the
  report as a breakdown and are explicitly not compared to a threshold.
- **Only methods that scored on EVERY pair are gate-eligible.** *Added beyond
  the review.* Pooling alone would otherwise reward a method for failing: a
  matcher that dies on the hardest pair has its pool drawn from the easy ones.
  Incomplete methods are marked `!` in the table and named in the verdict.
- **Recommend the simplest method that clears, not the lowest number.**
  Complexity tiers: similarity < homography (SIFT/ORB) < homography
  (LightGlue+DISK) < homography (LoFTR) < TPS. Ties *inside* a tier break on
  pooled median; that is the only place "lowest" picks anything.
- **Method 5 picks its base matcher by match-set properties** — grid coverage,
  then inlier ratio, then inlier count — never by control-point residual.
  Those are properties of the matches, not of the held-out truth, so using
  them costs nothing. Count is last because LoFTR returns ~8000 matches to
  SIFT's ~3800 and density is not quality.
- **The report says the bias out loud** on any GO: the figure is
  selected-best-of-N on a small single-subject sample and the honest read on
  new data is somewhat worse.

## 2026-09-11 — A percentile is refused when its sample cannot support it

`np.percentile(d_mm, 95)` on 10 values interpolates between the 9th and 10th
largest: it *is* the maximum. Printing that against a 3 mm threshold turns
"p95 < 3 mm" into "worst control point < 3 mm", far harsher than intended, and
one poorly-marked mole could fail the project.

The rule is `n * (1 - q/100) >= 1` — at least one sample above the percentile —
which puts p95 at n >= 20 and p90 at n >= 10. Below that the report prints
`p95 n/a (n=12, need 20)`. Every percentile is printed with its `n`.

**A third verdict exists: INCONCLUSIVE.** *Added beyond the review.* If a
method clears the median but p95 is unevaluable, that is not a GO. Declaring
GO on half a gate while refusing to print the other half would be having it
both ways.

## 2026-09-11 — The TPS dense warp is Newton-refined, not an independently fitted twin

`TPSPair` built its dense remap from `TPSTransform(dst, src)` — a separately
fitted reverse spline. Two regularised TPS fits are not inverses of one
another, so the blink compare was showing a slightly different transform than
the one the millimetres scored. The runbook calls blink compare the honest
test precisely because numbers can look fine while the warp is wrong, so it
must not be showing a different warp.

Option (a) of the review: the reverse fit is a SEED, and each grid point is
Newton-refined against the forward spline (analytic Jacobian, 3 iterations).
The round-trip residual `|fwd(inv(p)) - p|` is measured over the grid and
printed on the review screen in mm, alongside what the unrefined twin fit
would have been. On the synthetic fixtures the twin fit drifts 0.009–0.038 mm
and refinement drives it below 1e-4 mm.

## 2026-09-11 — Training labels are filtered by visibility; that rests on a lesion-size bound

`labels_in_tile` gains `min_visible` (default 0.5): a label is dropped from a
tile when the fraction of its disc inside the tile rect falls below it, so a
crop showing a sliver does not carry a label claiming a full-radius lesion.
Intersection area is computed EXACTLY (closed form, not disc sampling — a
sampled estimate's error scales with radius, so a fixed threshold would mean
different things for a 1.5 mm and a 5 mm lesion).

`assign_labels_to_tiles` is deliberately UNAFFECTED. Ownership stays
center-based: a count that changed with how much of a lesion a crop happened
to show would not be a count of lesions.

**This is only safe within a bound, and the bound is now computable.**
`max_safe_lesion_mm(px_per_mm, tile, overlap)` returns `(tile - stride) /
px_per_mm` — 128 px, or 4.9 mm at 26 px/mm, for the locked 640 px / 20%. A
tile shows a lesion of diameter d whole iff its centre is d/2 from both edges,
a window of width `tile - d`; consecutive windows start `stride` apart, so
they cover without a hole iff `d <= tile - stride`. Past that, visibility
filtering can delete a lesion from every tile. `lesion_size_warning()` exists
for the labeling tool to call at draw time.

**Correction to the review's premise, recorded because it changes how the knob
should be set.** Center-membership already requires the centre inside the
tile, and a disc whose centre lies in a half-plane always has >= half its area
in that half-plane. So a label "5 px from the edge with 90% of its disc
outside" cannot occur; against ONE edge the visible fraction cannot go below
0.5. The default 0.5 therefore prunes corner slivers only. Pruning EDGE
slivers needs `min_visible > 0.5` (0.7 drops a centre within ~0.35r of an
edge). 0.5 is kept as the default because it cannot delete a lesion the
overlap was meant to protect; the right value is an empirical question that
needs a validation set that does not exist yet.

## 2026-09-11 — Splits are write-once, and the boundary rule is a fixed trailing window

`save_split` overwrote `{name}.json` without a word. Rebuilding `v1` next
month would destroy the split last month's metrics were computed against —
the reproducibility this module exists to protect, defeated by its own
default.

- **Refuse to overwrite.** `SplitExistsError` names the existing file's
  `created_at` and counts and says to pick a new name. `--force` is the
  deliberate exception and prints what it destroys.
- **The filename carries a content hash**: `v1.a3f81c2d.json`, hashed over
  everything except `name` and `created_at`. A rebuild that changed something
  lands on a different filename; a rebuild that changed nothing lands on the
  same one, is recognised as the same split, and keeps its original
  `created_at`. A name can no longer silently mean two things; `load_split`
  errors rather than guessing if two files claim one name.
- **`load_split` re-runs the invariants.** Disjointness, chronological order,
  washout, count consistency, achieved ratios. A hand-edited or truncated file
  raises `SplitIntegrityError` instead of silently producing wrong metrics.
- **`achieved_ratios` is recorded alongside the requested `ratios`**, computed
  from day counts with washout days excluded from the denominator. A requested
  0.70/0.15/0.15 lands at 0.78/0.11/0.11 once the washout has eaten the front
  of val and test.

**The boundary rule is now fixed trailing calendar windows** (`mode="window"`,
the default): last 14 days test, the 14 before that val, everything earlier
train, washout between. A proportional test set is a different size every
build — too thin to measure anything early, needlessly large later; the
held-out set answers one fixed question and should be one fixed size, always
current. `mode="ratio"` is kept, named explicitly, for a retrospective split
over a finished capture period. Under the window rule a split needs about
`14 + 3 + 14 + 3 + 1 = 35` calendar days of capture; below that
`NotEnoughDataError` fires and says so.

Accepted consequence: for the first weeks train is smaller than test.
`format_report` prints a note saying that is the rule reporting honestly, not
a bug.

## 2026-09-11 — A pair no method could score is dropped before eligibility is judged

Follow-up to the pooled-gate entry above: the completeness rule it introduced
had a failure mode of its own.

If a single target cannot be scored by ANY method — usually because fewer than
three control-point ids are shared with the reference — then every method is
missing that pair, so every method is incomplete, so nothing is gate-eligible,
and the verdict reads **NO-GO**. That is a false NO-GO: a registration failure
announced because one image was under-marked. The breakdown table carried the
evidence (`pairs_scored/pairs_total`), but the verdict line is what gets read
at 11pm on day 4, and it would have been wrong.

`partition_pairs()` now splits pairs into scoreable and not, before anything
is decided. Unscoreable pairs are excluded from the pool, `pairs_total` counts
only the scoreable ones, and the report prints a block quote ABOVE the verdict
naming each excluded pair and why — "`left60_003.jpg` — only 2 control points
shared with the reference; scoring needs at least 3" — and saying plainly that
this is a data problem, not a registration result.

The asymmetry is deliberate and is what makes this safe: dropping such a pair
can only ever ADD eligible methods, never excuse one that genuinely failed. A
pair scored by SOME methods stays in the pool, and the methods that missed it
stay ineligible. Both directions are tested.
