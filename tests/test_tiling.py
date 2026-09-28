"""Tests for the tile coordinate convention.

These are cheap and they protect a convention that everything downstream
depends on. A one-pixel disagreement between the labeling tool and the
detector is invisible until metrics are mysteriously bad months later.

Runs standalone (no pytest needed):
    python -m tests.test_tiling
or under pytest if it is ever added to the venv.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tiling import (  # noqa: E402
    MIN_VISIBLE,
    OVERLAP,
    TILE_PX,
    Tile,
    assign_labels_to_tiles,
    covers_every_pixel,
    grid_shape,
    labels_in_tile,
    lesion_size_warning,
    max_safe_lesion_mm,
    owning_tile,
    stride_for,
    tiles_for,
    to_image_coords,
    to_tile_coords,
    visible_fraction,
)

SHAPES = [(3024, 4032), (4032, 3024), (640, 640), (1000, 640), (500, 300), (2000, 1500)]


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------

def test_stride_is_512_at_the_documented_defaults():
    assert stride_for(640, 0.2) == 512
    assert stride_for(TILE_PX, OVERLAP) == 512


def test_deterministic_across_calls():
    for shape in SHAPES:
        a = tiles_for(shape)
        b = tiles_for(shape)
        assert a == b, f"tiles_for({shape}) not deterministic"
        assert [t.tile_id for t in a] == [t.tile_id for t in b]


def test_tile_ids_are_derived_from_position_not_order():
    for shape in SHAPES:
        for t in tiles_for(shape):
            assert t.tile_id == f"r{t.row}c{t.col}"
    # Shuffling the list must not change what any id means.
    ts = tiles_for((2000, 1500))
    shuffled = ts[:]
    random.Random(0).shuffle(shuffled)
    for t in shuffled:
        assert t.tile_id == f"r{t.row}c{t.col}"


def test_grid_shape_matches_tiles_for():
    for shape in SHAPES:
        rows, cols = grid_shape(shape)
        ts = tiles_for(shape)
        assert len(ts) == rows * cols
        assert max(t.row for t in ts) == rows - 1
        assert max(t.col for t in ts) == cols - 1


def test_tiles_stay_inside_the_image_and_are_never_padded():
    for shape in SHAPES:
        h, w = shape
        for t in tiles_for(shape):
            assert 0 <= t.x0 < t.x1 <= w, f"{t} out of bounds in {shape}"
            assert 0 <= t.y0 < t.y1 <= h, f"{t} out of bounds in {shape}"
            # Clamped, never padded: a tile is at most TILE_PX, never more.
            assert t.width <= TILE_PX and t.height <= TILE_PX


def test_last_tile_is_clamped_to_the_edge():
    for shape in SHAPES:
        h, w = shape
        ts = tiles_for(shape)
        assert max(t.x1 for t in ts) == w, f"right edge not reached for {shape}"
        assert max(t.y1 for t in ts) == h, f"bottom edge not reached for {shape}"


def test_every_pixel_is_covered():
    for shape in SHAPES:
        ts = tiles_for(shape)
        assert covers_every_pixel(shape, ts), f"gap in coverage for {shape}"


def test_image_smaller_than_one_tile_yields_one_clipped_tile():
    ts = tiles_for((500, 300))
    assert len(ts) == 1
    t = ts[0]
    assert (t.x0, t.y0, t.x1, t.y1) == (0, 0, 300, 500)
    assert t.width == 300 and t.height == 500      # smaller than TILE_PX, by design


def test_membership_rect_is_half_open():
    t = Tile("r0c0", 0, 0, 100, 200, 740, 840)
    assert t.contains(100, 200)          # x0/y0 inclusive
    assert not t.contains(740, 300)      # x1 exclusive
    assert not t.contains(300, 840)      # y1 exclusive
    assert t.contains(739.999, 839.999)


# --------------------------------------------------------------------------
# coordinate round trip
# --------------------------------------------------------------------------

def test_round_trip_is_exact_for_integers():
    """Integer pixels are the persisted form, so this must be bit-exact."""
    rng = random.Random(1234)
    for shape in SHAPES:
        ts = tiles_for(shape)
        for _ in range(400):
            t = ts[rng.randrange(len(ts))]
            p = (rng.randrange(shape[1]), rng.randrange(shape[0]))
            back = to_image_coords(to_tile_coords(p, t), t)
            assert back == p, f"round trip lost integer {p} -> {back} on {t}"


def test_round_trip_for_floats_is_within_a_nanopixel():
    """Bit-exact is impossible in IEEE-754 - see to_tile_coords' docstring.

    Subtracting a multi-thousand integer offset drops low mantissa bits that
    adding it back cannot restore. 1e-9 px at ~26 px/mm is 4e-11 mm, against
    a 1.5 mm minimum lesion, so the bound below is astronomically tighter
    than anything this project measures.
    """
    rng = random.Random(1234)
    worst = 0.0
    for shape in SHAPES:
        ts = tiles_for(shape)
        for _ in range(400):
            t = ts[rng.randrange(len(ts))]
            p = (rng.uniform(0, shape[1]), rng.uniform(0, shape[0]))
            back = to_image_coords(to_tile_coords(p, t), t)
            worst = max(worst, abs(back[0] - p[0]), abs(back[1] - p[1]))
    assert worst < 1e-9, f"float round trip drifted by {worst} px"


def test_round_trip_from_tile_space_is_also_stable():
    """A click originates in tile space; it must survive the trip out."""
    rng = random.Random(99)
    ts = tiles_for((3024, 4032))
    for _ in range(400):
        t = ts[rng.randrange(len(ts))]
        q = (rng.uniform(0, t.width), rng.uniform(0, t.height))
        back = to_tile_coords(to_image_coords(q, t), t)
        assert abs(back[0] - q[0]) < 1e-9 and abs(back[1] - q[1]) < 1e-9


def test_to_tile_coords_of_tile_origin_is_zero():
    for t in tiles_for((3024, 4032)):
        assert to_tile_coords((t.x0, t.y0), t) == (0, 0)


# --------------------------------------------------------------------------
# label membership - the training/counting split
# --------------------------------------------------------------------------

def _labels(pts):
    return [{"id": i, "x_px": x, "y_px": y, "r_px": 8} for i, (x, y) in enumerate(pts)]


def test_labels_in_tile_uses_center_not_overlap():
    t = Tile("r0c0", 0, 0, 0, 0, 640, 640)
    # Centre outside, but a big radius spills in. Must NOT be included.
    spill = [{"id": 0, "x_px": 700.0, "y_px": 300.0, "r_px": 200}]
    assert labels_in_tile(spill, t) == []
    inside = [{"id": 1, "x_px": 639.0, "y_px": 300.0, "r_px": 200}]
    assert len(labels_in_tile(inside, t)) == 1


def test_a_label_in_the_overlap_band_is_in_several_tiles():
    """Documents the behaviour that makes the two functions necessary."""
    shape = (2000, 2000)
    ts = tiles_for(shape)
    # x=600 is inside tile col0 [0,640) and col1 [512,1152).
    lab = _labels([(600.0, 600.0)])
    hits = [t for t in ts if labels_in_tile(lab, t)]
    assert len(hits) > 1, "expected the overlap band to put one label in several tiles"


def test_labels_in_tile_drops_nothing():
    rng = random.Random(7)
    shape = (3024, 4032)
    ts = tiles_for(shape)
    lab = _labels(
        [(rng.uniform(0, shape[1] - 1), rng.uniform(0, shape[0] - 1)) for _ in range(500)]
    )
    seen: set[int] = set()
    for t in ts:
        for l in labels_in_tile(lab, t):
            seen.add(l["id"])
    assert seen == {l["id"] for l in lab}, "a label was visible in no tile at all"


def test_assign_labels_to_tiles_is_a_strict_partition():
    rng = random.Random(8)
    shape = (3024, 4032)
    ts = tiles_for(shape)
    lab = _labels(
        [(rng.uniform(0, shape[1] - 1), rng.uniform(0, shape[0] - 1)) for _ in range(500)]
    )
    assigned = assign_labels_to_tiles(lab, ts)

    flat = [l["id"] for v in assigned.values() for l in v]
    assert len(flat) == len(lab), "partition changed the label count"
    assert len(set(flat)) == len(lab), "a label was assigned to more than one tile"
    assert set(flat) == {l["id"] for l in lab}, "a label was dropped"


def test_owner_is_always_a_tile_that_actually_contains_the_label():
    rng = random.Random(9)
    shape = (2000, 1500)
    ts = tiles_for(shape)
    by_id = {t.tile_id: t for t in ts}
    lab = _labels(
        [(rng.uniform(0, shape[1] - 1), rng.uniform(0, shape[0] - 1)) for _ in range(300)]
    )
    for tile_id, ls in assign_labels_to_tiles(lab, ts).items():
        for l in ls:
            assert by_id[tile_id].contains(l["x_px"], l["y_px"])


def test_assignment_is_deterministic():
    rng = random.Random(10)
    shape = (2000, 2000)
    ts = tiles_for(shape)
    lab = _labels([(rng.uniform(0, 1999), rng.uniform(0, 1999)) for _ in range(200)])
    a = {k: [l["id"] for l in v] for k, v in assign_labels_to_tiles(lab, ts).items()}
    b = {k: [l["id"] for l in v] for k, v in assign_labels_to_tiles(lab, ts).items()}
    assert a == b


def test_label_outside_every_tile_raises_rather_than_vanishing():
    ts = tiles_for((1000, 1000))
    try:
        assign_labels_to_tiles(_labels([(5000.0, 5000.0)]), ts)
    except ValueError as e:
        assert "outside every tile" in str(e)
    else:
        raise AssertionError("a label outside the image was silently dropped")


def test_owning_tile_agrees_with_assign():
    rng = random.Random(11)
    shape = (2000, 1500)
    ts = tiles_for(shape)
    lab = _labels([(rng.uniform(0, 1499), rng.uniform(0, 1999)) for _ in range(200)])
    assigned = assign_labels_to_tiles(lab, ts)
    for tile_id, ls in assigned.items():
        for l in ls:
            assert owning_tile((l["x_px"], l["y_px"]), ts).tile_id == tile_id


# --------------------------------------------------------------------------
# visibility - a sliver is not a lesion
# --------------------------------------------------------------------------

def test_visible_fraction_matches_hand_computable_cases():
    t = Tile("r0c0", 0, 0, 0, 0, 640, 640)
    # Fully inside.
    assert abs(visible_fraction(320, 320, 50, t) - 1.0) < 1e-9
    # Centre exactly on the left edge: half the disc.
    assert abs(visible_fraction(0, 320, 50, t) - 0.5) < 1e-9
    # Centre exactly on a corner: a quarter.
    assert abs(visible_fraction(0, 0, 50, t) - 0.25) < 1e-9
    # A point label has no area to clip.
    assert visible_fraction(320, 320, 0, t) == 1.0


def test_visible_fraction_agrees_with_sampling_the_disc():
    """The exact formula against the obvious approximation, as a cross-check."""
    import math
    import random

    rng = random.Random(23)
    t = Tile("r0c0", 0, 0, 100, 100, 740, 740)
    for _ in range(40):
        cx = rng.uniform(50, 800)
        cy = rng.uniform(50, 800)
        r = rng.uniform(5, 200)
        exact = visible_fraction(cx, cy, r, t)
        hit = n = 0
        step = r / 40.0
        y = -r
        while y <= r:
            x = -r
            while x <= r:
                if x * x + y * y <= r * r:
                    n += 1
                    if t.contains(cx + x, cy + y):
                        hit += 1
                x += step
            y += step
        assert abs(exact - hit / n) < 0.02, (
            f"exact {exact:.3f} vs sampled {hit / n:.3f} at ({cx:.0f},{cy:.0f}) r={r:.0f}"
        )


def test_one_straight_edge_can_never_hide_more_than_half_a_lesion():
    """The geometry that decides what `min_visible` can possibly do.

    Center-membership already requires the centre to be inside the tile, and a
    disc whose centre is inside a half-plane always has at least half its area
    in that half-plane. So against ONE edge the visible fraction is >= 0.5 no
    matter how close the centre gets - "centred 5 px from the edge with 90% of
    the disc outside" cannot happen for a tile-resident label.

    A default of min_visible = 0.5 therefore prunes only corners. Raising it
    above 0.5 is what prunes edge slivers; see the module docstring.
    """
    t = Tile("r0c0", 0, 0, 0, 0, 640, 640)
    for r in (30.0, 100.0, 300.0):
        for x in (639.999, 639.0, 600.0):
            f = visible_fraction(x, 320.0, r, t)
            assert f >= 0.5 - 1e-9, f"r={r} x={x} gave {f}"
    assert abs(visible_fraction(639.9999, 320.0, 300.0, t) - 0.5) < 1e-3


def test_a_lesion_mostly_outside_the_tile_is_dropped():
    """The case the filter can actually catch at the default: a corner.

    Two edges at once, and the visible fraction goes to a quarter.
    """
    t = Tile("r0c0", 0, 0, 0, 0, 640, 640)
    sliver = [{"id": 0, "x_px": 5.0, "y_px": 5.0, "r_px": 100}]
    assert visible_fraction(5.0, 5.0, 100.0, t) < 0.3
    assert labels_in_tile(sliver, t) == [], (
        "a label three-quarters outside the crop was kept; that teaches the "
        "detector that a small arc is a full-radius lesion"
    )


def test_raising_min_visible_is_what_prunes_edge_slivers():
    """Documents the knob that implements the stated intent."""
    t = Tile("r0c0", 0, 0, 0, 0, 640, 640)
    half_cut = [{"id": 0, "x_px": 639.0, "y_px": 320.0, "r_px": 100}]
    assert len(labels_in_tile(half_cut, t)) == 1, "0.5 keeps a half-cut lesion"
    assert labels_in_tile(half_cut, t, min_visible=0.7) == []


def test_an_interior_tile_corner_drops_but_a_neighbour_keeps():
    """The drop is only acceptable because a neighbouring tile has it whole."""
    shape = (2000, 2000)
    ts = tiles_for(shape)
    # A seam corner well inside the image: x = y = 640 is the far corner of
    # tile r0c0 and sits in the overlap band of its neighbours.
    lab = [{"id": 0, "x_px": 636.0, "y_px": 636.0, "r_px": 40}]
    r0c0 = [t for t in ts if t.tile_id == "r0c0"][0]
    assert labels_in_tile(lab, r0c0) == [], "expected the corner sliver to drop"
    kept = [t for t in ts if labels_in_tile(lab, t)]
    assert kept, "the lesion was dropped from every tile"
    assert max(visible_fraction(636.0, 636.0, 40.0, t) for t in kept) > 0.999


def test_a_lesion_well_inside_the_tile_is_kept():
    t = Tile("r0c0", 0, 0, 0, 0, 640, 640)
    lab = [{"id": 0, "x_px": 320.0, "y_px": 320.0, "r_px": 100}]
    assert len(labels_in_tile(lab, t)) == 1


def test_min_visible_zero_restores_pure_center_membership():
    t = Tile("r0c0", 0, 0, 0, 0, 640, 640)
    sliver = [{"id": 0, "x_px": 639.0, "y_px": 320.0, "r_px": 300}]
    assert labels_in_tile(sliver, t, min_visible=0.0) == sliver


def test_labels_without_a_radius_are_never_dropped():
    t = Tile("r0c0", 0, 0, 0, 0, 640, 640)
    pts = [{"id": 0, "x_px": 639.0, "y_px": 639.0}]
    assert len(labels_in_tile(pts, t)) == 1
    zero = [{"id": 1, "x_px": 639.0, "y_px": 639.0, "r_px": 0}]
    assert len(labels_in_tile(zero, t)) == 1


def test_min_visible_is_validated():
    t = Tile("r0c0", 0, 0, 0, 0, 640, 640)
    for bad in (-0.1, 1.5):
        try:
            labels_in_tile([], t, min_visible=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted min_visible={bad}")


def test_a_dropped_lesion_is_still_whole_in_a_neighbouring_tile():
    """The filter is only safe because of this. Test the guarantee, not the hope.

    Every lesion up to the overlap bound must be retained by at least one
    tile, at full visibility, no matter where its centre lands.
    """
    shape = (2000, 2000)
    ts = tiles_for(shape)
    stride = stride_for(TILE_PX, OVERLAP)
    r = (TILE_PX - stride) / 2.0                  # the largest safe radius

    for x in range(int(r), 2000 - int(r), 37):    # sweep across seams
        lab = [{"id": 0, "x_px": float(x), "y_px": 1000.0, "r_px": r}]
        kept = [t for t in ts if labels_in_tile(lab, t)]
        assert kept, f"lesion at x={x} was dropped from every tile"
        best = max(visible_fraction(x, 1000.0, r, t) for t in kept)
        assert best > 0.999, (
            f"lesion at x={x} is shown whole by no tile (best {best:.3f})"
        )


def test_past_the_bound_a_lesion_can_be_shown_whole_by_nothing():
    """Documents why max_safe_lesion_mm has to be enforced upstream."""
    shape = (2000, 2000)
    ts = tiles_for(shape)
    stride = stride_for(TILE_PX, OVERLAP)
    r = (TILE_PX - stride)                        # twice the safe radius
    worst = min(
        max((visible_fraction(x, 1000.0, r, t) for t in ts if t.contains(x, 1000.0)),
            default=0.0)
        for x in range(int(r), 2000 - int(r), 13)
    )
    assert worst < 0.999, (
        "an oversized lesion was shown whole everywhere - the bound would be "
        "unnecessary, so either the bound or this test is wrong"
    )


def test_frame_edge_lesion_survives_when_the_image_shape_is_given():
    """No tile can show what is not in the photo.

    A lesion at the frame corner has three quarters of its disc outside the
    IMAGE. Measured against the whole disc it looks 75% occluded and gets
    dropped by every tile. Measured against its in-image area - which is all
    of the lesion that was ever photographed - it is fully visible.
    """
    shape = (2000, 2000)
    ts = tiles_for(shape)
    corner = [{"id": 0, "x_px": 1.0, "y_px": 1.0, "r_px": 60}]

    naive = [t for t in ts if labels_in_tile(corner, t)]
    assert not naive, "expected the plain-disc rule to drop a frame-corner lesion"

    aware = [t for t in ts if labels_in_tile(corner, t, image_shape=shape)]
    assert aware, "image_shape did not rescue a lesion at the frame corner"


def test_visibility_does_not_change_ownership():
    """assign_labels_to_tiles is unaffected - counting stays center-based."""
    shape = (2000, 2000)
    ts = tiles_for(shape)
    labs = [{"id": 0, "x_px": 639.0, "y_px": 320.0, "r_px": 300}]
    assigned = assign_labels_to_tiles(labs, ts)
    flat = [l["id"] for v in assigned.values() for l in v]
    assert flat == [0], "a barely-visible lesion vanished from the count"


def test_default_min_visible_is_one_half():
    assert MIN_VISIBLE == 0.5


# --------------------------------------------------------------------------
# the bound the visibility filter rests on
# --------------------------------------------------------------------------

def test_max_safe_lesion_is_the_overlap_in_mm():
    # 640 px tiles at 20% overlap -> stride 512 -> 128 px of overlap.
    assert max_safe_lesion_mm(1.0) == 128.0
    assert abs(max_safe_lesion_mm(26.0) - 128.0 / 26.0) < 1e-12
    # The documented number for the locked geometry.
    assert abs(max_safe_lesion_mm(26.0) - 4.923) < 0.001


def test_max_safe_lesion_scales_with_overlap():
    assert max_safe_lesion_mm(26.0, 640, 0.4) > max_safe_lesion_mm(26.0, 640, 0.2)
    assert max_safe_lesion_mm(26.0, 640, 0.0) == 0.0


def test_max_safe_lesion_rejects_a_nonsense_scale():
    for bad in (0.0, -3.0):
        try:
            max_safe_lesion_mm(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted px_per_mm={bad}")


def test_lesion_size_warning_fires_only_past_the_bound():
    assert lesion_size_warning(3.0, 26.0) is None
    assert lesion_size_warning(4.9, 26.0) is None
    msg = lesion_size_warning(8.0, 26.0)
    assert msg and "8.0" in msg and "4.9" in msg, msg


def test_the_bound_matches_what_the_tiles_actually_do():
    """Derivation and implementation, checked against each other."""
    for tile, overlap in ((640, 0.2), (512, 0.25), (800, 0.1)):
        stride = stride_for(tile, overlap)
        assert max_safe_lesion_mm(1.0, tile, overlap) == tile - stride


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
    print("tiling")
    print("-" * 62)
    raise SystemExit(_run())
