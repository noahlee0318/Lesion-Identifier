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
    OVERLAP,
    TILE_PX,
    Tile,
    assign_labels_to_tiles,
    covers_every_pixel,
    grid_shape,
    labels_in_tile,
    owning_tile,
    stride_for,
    tiles_for,
    to_image_coords,
    to_tile_coords,
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
