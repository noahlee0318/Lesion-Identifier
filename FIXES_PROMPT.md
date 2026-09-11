# Review fixes — register.py, tiling.py, splits.py

Findings from an outside review of the code as of 2026-09-11. Read
`docs/DECISIONS.md` first; it is new and supersedes the morning snapshots in
`docs/runbook.md` and `docs/build-plan.md` where they disagree.

Two of the spec errors you caught earlier (center-membership not partitioning,
float round-trip not being bit-exact) were real, and you were right to push
back rather than implement them quietly. Do the same here. Anything below that
you think is wrong, say so before changing it — a wrong "fix" to the gate
statistics is worse than the current state, because it would be wrong in a way
that looks reviewed.

Work in the order below. Items 1 and 2 gate whether the phase 0 number can be
believed at all; everything after is quality.

---

## 1. register.py — the gate number is optimistically biased

Two compounding problems, both selection-on-the-test-set:

- `write_report` sorts all six methods by median and reports the minimum as the
  headline GO/NO-GO figure.
- Method 5 picks its base matcher by control-point median, then gets scored on
  those same control points.

With ~10 control points per pair and six candidates, the minimum is meaningfully
below the true out-of-sample error. This is the number that decides whether the
project proceeds, so it must not flatter itself.

Fix:

- **Pool residuals across all pairs, per method**, and compute median / p90 /
  p95 / max on the pooled vector. Do not take a median-of-per-pair-medians; that
  is a different and weaker statistic, and it hides a single catastrophic pair.
  Keep per-pair numbers in the report as a breakdown, but the gate reads from
  the pool.
- **Choose the winner by a rule that is not "lowest number".** Report every
  method that clears the gate, and recommend the SIMPLEST one that clears it
  (order: similarity < homography < TPS; within homography: SIFT/ORB <
  LightGlue < LoFTR). A method that wins by 0.05mm on 50 points has not earned
  the extra machinery. State the recommendation and the reason in the report.
- **Say the bias out loud in the report.** One line: the winning method's figure
  is selected-best-of-N on a small sample and is therefore optimistic; the
  honest read on new data is somewhat worse.
- For method 5 specifically: pick its base matcher by **inlier count and match
  distribution**, not by control-point median. Those are properties of the match
  set, not of the held-out truth, so using them costs nothing.

If you disagree with pooling — e.g. you think per-pair variance matters more —
argue it before changing anything.

## 2. register.py — p95 on ~10 points is just the maximum

`np.percentile(d_mm, 95)` on 10 values interpolates between the 9th and 10th
largest, so the "p95 < 3mm" half of the gate is effectively "worst control point
< 3mm". That is far harsher than intended, and one poorly-marked mole can fail
the project.

Pooling (item 1) mostly fixes this — five pairs × ~10 points is ~50 residuals
and p95 becomes meaningful. Additionally:

- **Report `n` next to every percentile.** A percentile without its sample size
  is not interpretable.
- **Refuse to print a p95 computed on fewer than 20 pooled residuals.** Print
  `p95 n/a (n=12, need 20)` instead. A number that cannot mean what it claims
  should not appear next to a gate threshold.

## 3. register.py — `_spatial_subsample` discards its own work

The function grid-thins matches for even coverage, then ends with:

    keep = np.array(sorted(keep))[:target]

which truncates by array index — detector output order, not position. The
spatial evenness is thrown away on the last line, and TPS gets clumped points
anyway.

Fix: select round-robin across occupied cells until `target` is reached, so the
truncation is itself spatially even. Add a test: given matches synthetically
clumped into one corner plus a sparse spread elsewhere, the subsample must
retain points from a comparable number of distinct cells rather than mostly the
clump.

## 4. register.py — the TPS inverse is not the inverse

`inv = TPSTransform(sb, sa, lam=tps_lambda, ...)` is an independently fitted
reverse map. Two regularised TPS fits are not inverses of one another; with
`lam=1e-3` both are deliberately smoothed, and the composition drifts.

The consequence is specific and bad: `TPSPair.warp_to_reference` uses `inv` to
build the dense remap, so the **blink compare displays a slightly different
transform than the one the numbers score.** The runbook calls blink compare the
honest test precisely because numbers can look fine while the warp is wrong —
so it must not be showing a different warp.

Pick one, and say which in a docstring:

- **(a)** Newton-refine the inverse: for each grid point, seed with `inv`, then
  iterate `q <- q - J^-1 (fwd(q) - p)` two or three times. Report the max
  residual of `fwd(inv(p)) - p` over the grid, in mm, on the review screen.
- **(b)** Keep the twin fit but **measure and display** the same round-trip
  residual, so the reviewer knows how much the picture and the number differ.

(a) is better if the cost is acceptable; (b) is acceptable only with the
readout. Silently keeping the current behaviour is not.

## 5. tiling.py — no visibility filter on training labels

`labels_in_tile` returns every label centred in the tile, including one centred
5px from the edge whose disc is 90% outside. The crop shows a sliver; the label
claims a full-radius lesion. That teaches the detector that a small arc is a
large lesion.

Add a `min_visible: float = 0.5` parameter: drop a label from a tile when the
fraction of its disc inside the tile rect falls below it. Circle-rectangle
intersection area is fine to approximate — sample the disc on a small grid, or
use the exact formula if you prefer; document which.

This is only safe because the overlap guarantees a neighbouring tile shows the
lesion whole, and **that guarantee has a bound**: it holds while lesion diameter
< overlap in pixels. At 640px / 20% that is 128px, which at 26 px/mm is 4.9mm.
Add a module-level function `max_safe_lesion_mm(px_per_mm, tile, overlap)` and
have the labeling tool warn when a lesion is sized past it. Document the bound
in the MEMBERSHIP section of the module docstring.

`assign_labels_to_tiles` is unaffected — ownership stays center-based.

Also: the `to_tile_coords` docstring gives the float residual as ~1e-9 px in the
guarantee block and ~1e-12 px in the prose below it. Make them agree.

## 6. splits.py — `save_split` silently overwrites

`build_split(persist=True)` → `save_split` → `p.write_text(...)` overwrites
`{name}.json` without a word. Rebuild `v1` next month with more data and the
split that last month's metrics were computed against is gone. That is exactly
the reproducibility this module exists to protect, defeated by its own default.

- **Refuse to overwrite an existing split file.** Raise with the existing file's
  `created_at` and counts, and tell the caller to pick a new name.
- Add `--force` for the deliberate case, which must print what it is destroying.
- Consider appending a short content hash to the filename (`v1.a3f81c.json`) so
  a name can never silently mean two different things.

## 7. splits.py — record achieved ratios, and validate on load

- `Split.ratios` stores what was requested. After a 3-day washout, a requested
  0.70/0.15/0.15 lands near 0.78/0.11/0.11. Add `achieved_ratios`, computed from
  day counts, so the JSON is self-explanatory six months from now.
- `load_split` parses JSON and returns it unchecked. Re-run the invariants on
  load — disjointness, chronological order, washout respected. A hand-edited or
  truncated split file should fail loudly, not silently produce wrong metrics.

## 8. splits.py — replace the ratio boundary with a fixed recent window

Proportional splits make the test set grow without bound: thin and unstable
early, needlessly large later. What you want is a stable-size, always-current
held-out set.

Change the default boundary rule to **fixed trailing windows**: last `test_days`
(default 14) is test, the `val_days` (default 14) before that is val, everything
earlier is train, with the existing calendar washout between blocks. Keep
`plan_split`'s ratio mode available as an explicitly-named alternative — do not
delete it — but make the fixed-window rule the default and say why in the
docstring.

Under this rule, `NotEnoughDataError` fires until roughly
`14 + 3 + 14 + 3 + 1 = 35` days of capture, which is correct and should say so
in the same encouraging-but-honest tone the current message has.

## 9. Minor

- `refined()` proceeds with no coarse step when Face Mesh fails, but the result
  is still named `coarse+SIFT`. Set `res.note` to record that the coarse step
  was skipped, and reflect it in the name.
- `match_loftr` loads `pretrained="outdoor"`. Try `"indoor"` as well — one line,
  plausibly a better prior for close-range faces — and keep whichever wins on
  the pooled metric. Make it a parameter, not a constant.
- `run_pair` raises `SystemExit` on a missing fiducial. That kills batch runs and
  makes the function untestable. Raise a `RegistrationInputError` and let the
  CLI turn it into an exit code.

---

## After

Re-run the full test suite plus a synthetic end-to-end pass, then give me a
short report: what changed, what you disagreed with and why, and the current
pooled numbers on synthetic data so I can sanity-check the reporting path before
real photos exist.

Do not start the labeling tool until these are reviewed.
