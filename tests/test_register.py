"""Tests for the parts of register.py that decide the phase 0 gate.

These do not run a matcher or touch a photo. They test the things that would
silently corrupt the GO/NO-GO number: how residuals are summarised, which
method gets recommended, whether a percentile is allowed to be printed, and
whether the warp shown in the blink compare is the warp being scored.

Runs standalone (no pytest needed):
    python -m tests.test_register
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, register as R  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _result(name: str, residuals_mm, *, kind="homography", matches=1000,
            inliers=900, coverage=0.5, ratio=0.9) -> R.MethodResult:
    """A MethodResult carrying a chosen residual vector, as if it had scored."""
    v = np.asarray(residuals_mm, float)
    r = R.MethodResult(name, kind)
    r.ok = True
    r.residuals_mm = v
    r.median_mm = float(np.median(v))
    r.p90_mm = float(np.percentile(v, 90))
    r.p95_mm = float(np.percentile(v, 95))
    r.max_mm = float(v.max())
    r.n_matches, r.n_inliers = matches, inliers
    r.cell_coverage, r.inlier_ratio = coverage, ratio
    r.seconds = 1.0
    return r


def _pair(target: str, results: list[R.MethodResult]) -> dict:
    return {
        "reference": Path("ref.jpg"),
        "target": Path(target),
        "px_per_mm": 26.0,
        "ref_shape": (3000, 2000),
        "n_control_points": 10,
        "results": results,
    }


# --------------------------------------------------------------------------
# spatial subsampling - even coverage, all the way through the truncation
# --------------------------------------------------------------------------

def _clumped(n_clump=4000, n_spread=600, size=3000):
    """Most matches piled into one corner, a sparse spread over the rest."""
    rng = np.random.default_rng(1)
    clump = rng.uniform(0, size * 0.1, (n_clump, 2))
    spread = rng.uniform(0, size, (n_spread, 2))
    pb = np.vstack([clump, spread])
    return pb.copy(), pb


def test_subsample_keeps_points_from_the_whole_frame_not_the_clump():
    shape = (3000, 3000)
    pa, pb = _clumped()
    g = max(1, int(np.sqrt(500)))
    cells_before = len(np.unique(R._cell_keys(pb, shape, g)))

    sa, sb = R._spatial_subsample(pa, pb, shape, target=500)
    cells_after = len(np.unique(R._cell_keys(sb, shape, g)))

    assert cells_after == cells_before, (
        f"thinning lost coverage: {cells_before} occupied cells -> {cells_after}"
    )
    in_clump = float((sb.max(axis=1) < 300).mean())
    assert in_clump < 0.25, (
        f"{in_clump:.0%} of the thinned points are in the clump; the clump is "
        "87% of the input, so the thinning is not doing its job"
    )


def test_subsample_actually_returns_the_number_of_points_asked_for():
    """The per-cell-quota version returned at most one point per cell.

    With a 22x22 grid that is at most 484 of the 500 requested, and the spline
    is capped at the thinning grid's resolution no matter how many good
    matches exist.
    """
    pa, pb = _clumped()
    sa, sb = R._spatial_subsample(pa, pb, (3000, 3000), target=500)
    assert len(sa) == 500, f"asked for 500 well-spread points, got {len(sa)}"
    assert len(sb) == 500


def test_subsample_truncation_stays_even_on_a_fine_grid():
    """Guards the latent case: more cells than `target`.

    Here the quota collapses to one per cell and there are more occupied cells
    than the target, so something must be dropped. Dropping by array index -
    detector output order - would take a contiguous run of the frame. Dropping
    round-robin cannot.
    """
    shape = (3000, 3000)
    rng = np.random.default_rng(4)
    pb = rng.uniform(0, 3000, (5000, 2))
    pa = pb.copy()
    # A grid far finer than sqrt(target): 60x60 = 3600 cells for 300 points.
    keys = R._cell_keys(pb, shape, 60)
    order = np.argsort(keys)                      # worst case: sorted by cell
    pa, pb = pa[order], pb[order]

    sa, sb = R._spatial_subsample(pa, pb, shape, target=300)
    assert len(sb) == 300
    # Spread over the frame, not one horizontal band of it.
    assert np.ptp(sb[:, 1]) > 0.8 * 3000, "kept points cover only part of the frame"
    assert np.ptp(sb[:, 0]) > 0.8 * 3000


def test_subsample_keeps_the_two_point_sets_aligned():
    pa, pb = _clumped()
    pa = pa + np.array([1000.0, 0.0])              # make them distinguishable
    sa, sb = R._spatial_subsample(pa, pb, (3000, 3000), target=500)
    assert np.allclose(sa - sb, np.array([1000.0, 0.0])), (
        "subsampling desynchronised the target/reference point correspondence"
    )


def test_subsample_is_deterministic():
    pa, pb = _clumped()
    a1, b1 = R._spatial_subsample(pa, pb, (3000, 3000), target=500)
    a2, b2 = R._spatial_subsample(pa, pb, (3000, 3000), target=500)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)


# --------------------------------------------------------------------------
# the TPS inverse is an inverse
# --------------------------------------------------------------------------

def _tps_pair(lam=1e-3, size=3000, n=120):
    rng = np.random.default_rng(0)
    src = rng.uniform(0, size, (n, 2))

    def warp(p):
        x, y = p[:, 0], p[:, 1]
        return np.column_stack([x + 40 * np.sin(y / 900.0), y + 30 * np.cos(x / 1100.0)])

    dst = warp(src)
    diag = float(np.hypot(size, size))
    fwd = R.TPSTransform(src, dst, lam=lam, scale=diag)
    inv = R.TPSTransform(dst, src, lam=lam, scale=diag)
    return fwd, inv


def test_tps_jacobian_matches_finite_differences():
    fwd, _ = _tps_pair()
    rng = np.random.default_rng(3)
    q = rng.uniform(500, 2500, (6, 2))
    s = fwd.scale
    _, J = fwd._map_and_jac_chunk(q / s)
    h = 1e-6
    for k in range(len(q)):
        num = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            plus = fwd._map_chunk(((q[k] + e) / s)[None])[0]
            minus = fwd._map_chunk(((q[k] - e) / s)[None])[0]
            num[:, j] = (plus - minus) / (2 * h / s)
        assert np.allclose(num, J[k], atol=1e-5), f"Jacobian wrong at {q[k]}"


def test_newton_inverse_beats_the_twin_fit_by_orders_of_magnitude():
    """The twin fit is a seed, not an inverse.

    Two regularised splines fitted in opposite directions are not inverses of
    one another, so the dense warp built from the reverse fit is a different
    transform than the one the millimetres score.
    """
    fwd, inv = _tps_pair()
    grid = np.array(
        [[x, y] for x in np.linspace(300, 2700, 20) for y in np.linspace(300, 2700, 20)],
        float,
    )
    seed = inv.map_points(grid)
    twin_err = np.linalg.norm(fwd.map_points(seed) - grid, axis=1).max()

    refined = fwd.invert_points(grid, seed, iters=3)
    newton_err = np.linalg.norm(fwd.map_points(refined) - grid, axis=1).max()

    assert twin_err > 1e-4, "twin fit was already exact - test is not exercising anything"
    assert newton_err < twin_err / 100.0, (
        f"Newton refinement did not converge: {twin_err:.3e} px -> {newton_err:.3e} px"
    )
    assert newton_err < 1e-6


def test_tps_warp_records_its_round_trip_error():
    fwd, inv = _tps_pair()
    pair = R.TPSPair(fwd, inv)
    img = np.zeros((600, 400, 3), np.uint8)
    pair.warp_to_reference(img, (600, 400))
    rt = pair.last_roundtrip
    assert rt is not None, "the warp did not report how far it is from the scored map"
    for k in ("max_px", "p95_px", "median_px", "max_in_hull_px", "newton_iters"):
        assert k in rt, f"round-trip readout is missing {k}"
    assert rt["n_inside_hull"] <= rt["n_grid"]


def test_homography_reports_no_round_trip_error():
    """A homography's dense inverse is algebraic, so there is nothing to warn about."""
    H = np.array([[1.02, 0.01, 5.0], [-0.01, 0.99, -3.0], [1e-6, 2e-6, 1.0]])
    T = R.HomographyTransform(H)
    assert T.last_roundtrip is None


# --------------------------------------------------------------------------
# percentiles refuse to overstate their sample
# --------------------------------------------------------------------------

def test_min_n_for_percentile():
    assert R.min_n_for_percentile(95) == 20
    assert R.min_n_for_percentile(90) == 10
    assert R.min_n_for_percentile(50) == 2


def test_p95_is_refused_below_twenty_samples():
    v = np.linspace(0.1, 2.0, 12)
    assert R.pct_or_none(v, 95) is None, (
        "p95 on 12 points is the interpolation between the top two values - "
        "it must not be printed next to a gate threshold"
    )
    assert R.pct_or_none(np.linspace(0.1, 2.0, 20), 95) is not None


def test_refused_percentile_prints_its_sample_size_and_requirement():
    text = R.fmt_pct(None, 12, 95)
    assert "n/a" in text and "n=12" in text and "20" in text, text


def test_every_printed_percentile_carries_n():
    """The pooled table must never show a percentile without its sample size."""
    pairs = [_pair("t1.jpg", [_result(R.M1, np.linspace(0.2, 1.0, 10))])]
    out = Path(config.REPO) / "reports" / "_test_register_report.md"
    R.write_report(pairs, out, [])
    body = out.read_text(encoding="utf-8")
    out.unlink()
    assert "| n |" in body, "pooled table has no n column"
    assert "n/a (n=10, need 20)" in body, (
        "a p95 on 10 pooled residuals was printed as if it were a p95"
    )


# --------------------------------------------------------------------------
# pooling, not median-of-medians
# --------------------------------------------------------------------------

def test_pooling_does_not_hide_one_catastrophic_pair():
    good = np.full(10, 0.5)
    awful = np.full(10, 9.0)
    pairs = [
        _pair("t1.jpg", [_result(R.M1, good)]),
        _pair("t2.jpg", [_result(R.M1, good)]),
        _pair("t3.jpg", [_result(R.M1, awful)]),
    ]
    pooled = {m.name: m for m in R.pool_by_method(pairs)}[R.M1]

    median_of_medians = float(np.median([0.5, 0.5, 9.0]))
    assert median_of_medians == 0.5
    assert pooled.n == 30
    assert pooled.max_mm == 9.0
    assert pooled.p90_mm is not None and pooled.p90_mm > 5.0, (
        "pooled p90 should see the bad pair; a median of per-pair medians does not"
    )
    assert pooled.worst_pair == "t3.jpg", "the worst pair is not named"


def test_pooled_n_is_the_total_residual_count():
    pairs = [_pair(f"t{i}.jpg", [_result(R.M1, np.linspace(0.1, 1.0, 10))]) for i in range(5)]
    pooled = R.pool_by_method(pairs)[0]
    assert pooled.n == 50 and pooled.pairs_scored == 5


def test_method_five_pools_under_one_name_even_with_different_bases():
    """Method 5 picks its base per pair. If the base went in the NAME, one
    method would split into several one-pair rows and never be gate-eligible."""
    a = _result(R.M5, np.full(10, 0.4), kind="tps")
    a.note = f"base: {R.M1}"
    b = _result(R.M5, np.full(10, 0.6), kind="tps")
    b.note = f"base: {R.loftr_name('indoor')}"
    pooled = R.pool_by_method([_pair("t1.jpg", [a]), _pair("t2.jpg", [b])])
    assert len(pooled) == 1, "method 5 split into several rows"
    assert pooled[0].n == 20 and pooled[0].complete


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def _clearing(name, value, n=25):
    return _result(name, np.full(n, value))


def test_gate_recommends_the_simplest_clearing_method_not_the_lowest_number():
    pairs = [
        _pair(
            "t1.jpg",
            [
                _clearing(R.M1, 0.90),                    # homography, classical
                _clearing(R.M3, 0.88),                    # learned detector
                _clearing(R.M5, 0.85),                    # TPS - lowest number
            ],
        )
    ]
    v = R.decide_gate(R.pool_by_method(pairs))
    assert v.status == "GO"
    assert v.recommended.name == R.M1, (
        f"recommended {v.recommended.name}; a 0.05 mm win does not buy a "
        "thin-plate spline"
    )
    assert len(v.clearing) == 3, "every clearing method should be reported"


def test_gate_falls_through_to_a_harder_method_when_simpler_ones_fail():
    pairs = [
        _pair(
            "t1.jpg",
            [
                _clearing(R.M0, 4.0),                     # similarity: fails
                _clearing(R.M1, 2.0),                     # homography: fails median
                _clearing(R.M5, 0.9),                     # TPS: clears
            ],
        )
    ]
    v = R.decide_gate(R.pool_by_method(pairs))
    assert v.status == "GO" and v.recommended.name == R.M5


def test_in_tier_ties_break_on_the_pooled_median():
    pairs = [_pair("t1.jpg", [_clearing(R.M2, 0.5), _clearing(R.M1, 0.9)])]
    v = R.decide_gate(R.pool_by_method(pairs))
    # SIFT and ORB share a tier, so here - and only here - the number decides.
    assert v.recommended.name == R.M2


def test_a_method_that_missed_a_pair_cannot_set_the_gate():
    """Otherwise a method is rewarded for failing on the hard pair."""
    failed = R.MethodResult(R.M3, "homography")
    failed.note = "matcher failed"
    pairs = [
        _pair("easy.jpg", [_clearing(R.M1, 1.2), _clearing(R.M3, 0.2)]),
        _pair("hard.jpg", [_clearing(R.M1, 1.2), failed]),
    ]
    v = R.decide_gate(R.pool_by_method(pairs))
    assert v.recommended.name == R.M1
    assert [m.name for m in v.ineligible] == [R.M3]
    assert R.M3 not in [m.name for m in v.clearing]


def test_median_clears_but_p95_unevaluable_is_inconclusive_not_go():
    """Half a gate is not a gate."""
    pairs = [_pair("t1.jpg", [_result(R.M1, np.full(10, 0.5))])]
    v = R.decide_gate(R.pool_by_method(pairs))
    assert v.status == "INCONCLUSIVE", v.status
    assert v.recommended is not None and not v.clearing


def test_gate_fails_when_nothing_clears():
    pairs = [_pair("t1.jpg", [_clearing(R.M1, 4.0), _clearing(R.M5, 3.9)])]
    v = R.decide_gate(R.pool_by_method(pairs))
    assert v.status == "NO-GO" and v.recommended is None


def test_report_states_the_selection_bias_on_a_go():
    pairs = [_pair("t1.jpg", [_clearing(R.M1, 0.5), _clearing(R.M5, 0.4)])]
    out = Path(config.REPO) / "reports" / "_test_register_bias.md"
    R.write_report(pairs, out, [])
    body = out.read_text(encoding="utf-8")
    out.unlink()
    assert "optimistic" in body.lower(), "the report does not admit the selection bias"
    assert "pooled" in body.lower()


# --------------------------------------------------------------------------
# method 5 picks its base without touching the held-out truth
# --------------------------------------------------------------------------

def test_tps_base_is_chosen_on_coverage_not_on_control_point_error():
    dummy = (np.zeros((10, 2)), np.zeros((10, 2)))
    # `wide` has a worse control-point median but much better coverage.
    wide = _result(R.M1, np.full(10, 1.2), inliers=800, coverage=0.9, ratio=0.85)
    narrow = _result(R.M3, np.full(10, 0.2), inliers=800, coverage=0.3, ratio=0.99)
    picked = R._pick_tps_base([(wide, dummy, np.eye(3)), (narrow, dummy, np.eye(3))])
    assert picked[0].name == R.M1, (
        "the TPS base was picked using held-out control points again"
    )


def test_tps_base_ignores_matchers_with_too_few_inliers():
    dummy = (np.zeros((10, 2)), np.zeros((10, 2)))
    thin = _result(R.M1, np.full(10, 0.1), inliers=4, coverage=1.0, ratio=1.0)
    fat = _result(R.M3, np.full(10, 1.0), inliers=900, coverage=0.6, ratio=0.9)
    picked = R._pick_tps_base([(thin, dummy, np.eye(3)), (fat, dummy, np.eye(3))])
    assert picked[0].name == R.M3


def test_tps_base_returns_nothing_when_no_matcher_is_usable():
    thin = _result(R.M1, np.full(10, 0.1), inliers=2)
    assert R._pick_tps_base([(thin, None, np.eye(3))]) == ()


def test_raw_inlier_count_does_not_outrank_coverage():
    """LoFTR returns ~8000 matches to SIFT's ~3800. Density is not quality."""
    dummy = (np.zeros((10, 2)), np.zeros((10, 2)))
    dense = _result(R.loftr_name("outdoor"), np.full(10, 1.0),
                    matches=9000, inliers=8000, coverage=0.4, ratio=0.89)
    spread = _result(R.M1, np.full(10, 1.0),
                     matches=4000, inliers=3800, coverage=0.9, ratio=0.95)
    picked = R._pick_tps_base([(dense, dummy, np.eye(3)), (spread, dummy, np.eye(3))])
    assert picked[0].name == R.M1


# --------------------------------------------------------------------------
# input errors are errors, not exits
# --------------------------------------------------------------------------

def test_missing_image_raises_instead_of_killing_the_process():
    try:
        R.run_pair(Path("no_such_reference.jpg"), Path("no_such_target.jpg"))
    except R.RegistrationInputError as e:
        assert "could not read" in str(e)
    except SystemExit:
        raise AssertionError(
            "run_pair raised SystemExit; that kills a batch run and makes the "
            "function untestable"
        )
    else:
        raise AssertionError("a missing image did not raise")


def test_registration_input_error_is_not_a_systemexit():
    assert not issubclass(R.RegistrationInputError, SystemExit)


def test_loftr_prior_is_a_parameter_not_a_constant():
    import inspect

    assert "pretrained" in inspect.signature(R.match_loftr).parameters
    assert set(R.LOFTR_WEIGHTS) == {"indoor", "outdoor"}
    assert R.loftr_name("indoor") != R.loftr_name("outdoor")


def test_method_names_are_stable_pooling_keys():
    """Every ladder name must map to a known complexity tier."""
    for name in (R.M0, R.M1, R.M2, R.M3, R.M5,
                 R.loftr_name("indoor"), R.loftr_name("outdoor")):
        assert R.method_tier(name) in R.TIER_LABEL, name
    assert R.method_tier(R.M0) < R.method_tier(R.M1) < R.method_tier(R.M3)
    assert R.method_tier(R.M3) < R.method_tier(R.loftr_name("indoor"))
    assert R.method_tier(R.loftr_name("indoor")) < R.method_tier(R.M5)
    assert R.method_tier(R.M1) == R.method_tier(R.M2)


# --------------------------------------------------------------------------

def _run() -> int:
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")
    print("-" * 62)
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    print("register")
    print("-" * 62)
    raise SystemExit(_run())
