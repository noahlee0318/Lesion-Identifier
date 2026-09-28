"""Tile geometry - the single shared definition of how an image is cut.

Both the labeling tool and (later) the detector import from here. Two
implementations that disagree by even one pixel would put labels in a
different coordinate space than the crops they describe, and that surfaces
months later as inexplicably bad metrics with no obvious cause.

=============================================================================
COORDINATE CONTRACT

Labels are stored in IMAGE coordinates. Always. Tiles are a VIEW, generated
on demand, never a storage format.

Storing labels per-tile would mean that changing `tile` or `overlap` later
invalidates every label ever made - months of annotation destroyed by a
hyperparameter change. In image coordinates, retiling is free.

The labeling tool converts a click from tile coordinates to image coordinates
before anything is persisted. If a tile-relative coordinate is being written
anywhere outside a transient UI variable, that is a bug.
=============================================================================

EDGE HANDLING

Tiles step by `stride = round(tile * (1 - overlap))` (640 @ 20% -> 512). The
last tile in a row or column will not land evenly. It is CLAMPED to the image
edge and the resulting larger overlap is accepted.

It is deliberately NOT zero-padded. Padding teaches a detector that black
borders are a real image feature, and since every image is padded on the same
two edges, that artefact correlates with position in the frame - which is
exactly the kind of spurious signal a single-subject dataset will latch onto.

The detector must inherit this choice: run sliced inference with the same
clamped grid, and merge with NMS across the enlarged seams.

If an image is SMALLER than one tile on an axis, a single tile is emitted,
clipped to the image. Such a tile is smaller than `tile` px - callers that
need a fixed input size must resize or letterbox at that point, not here.

MEMBERSHIP

A lesion belongs to a tile if its CENTER is inside the tile's half-open rect
[x0, x1) x [y0, y1). Not "any overlap" - lesions are small and roughly
point-like, and overlap-membership would duplicate every lesion near a seam
into two label sets with two different truncated extents.

Because tiles overlap, center-membership does NOT partition the labels: a
center inside the overlap band is inside two tiles (four at a corner). That
is correct for training and wrong for counting, so the two cases are separate
functions:

    labels_in_tile()         every label centered in this tile AND at least
                             `min_visible` of its disc inside it.
                             A label may appear in several tiles.
                             USE FOR TRAINING - a tile that renders a lesion
                             must carry its label, or the crop teaches the
                             detector that a visible lesion is background.

    assign_labels_to_tiles() exactly one owning tile per label. UNAFFECTED by
                             visibility - ownership stays purely center-based.
                             USE FOR COUNTING AND COVERAGE.
                             Owner = tile whose centre is nearest the lesion
                             (most surrounding context), ties broken by
                             (row, col) so it is deterministic.

This mirrors `detections.owned` for the pose overlap band: kept everywhere it
is visible, counted in exactly one place.

VISIBILITY, AND THE LESION-SIZE BOUND IT RESTS ON

Center-membership alone admits a lesion whose disc is mostly outside the tile.
The crop shows a sliver; the label claims a full-radius lesion. That teaches
the detector that a small arc is a large lesion, which is worse than not
showing it the lesion at all. So `labels_in_tile` drops a label whose visible
disc fraction falls below `min_visible` (default 0.5).

READ THIS BEFORE TUNING `min_visible`. The centre is already required to be
inside the tile, and a disc whose centre lies in a half-plane always has at
least HALF its area in that half-plane. So against a single tile edge the
visible fraction cannot go below 0.5, however close the centre gets: a label
"5 px from the edge with 90% of its disc outside" is not a thing that can
happen here. Only a CORNER, cutting on two edges at once, reaches down toward
0.25.

The practical consequence:

    min_visible = 0.5   (default)  prunes corner slivers only. A lesion cut
                                   exactly in half by one edge is KEPT.
    min_visible > 0.5              is what prunes edge slivers. 0.7 drops a
                                   centre within roughly 0.35r of one edge.

0.5 is the default because it is the value that cannot delete a lesion the
overlap was supposed to protect, and because the right number for a detector
is an empirical question that needs labels that do not exist yet. Raise it
once there is a validation set to measure it against. Dropping is only safe because the overlap
guarantees some OTHER tile shows the same lesion whole -

    AND THAT GUARANTEE HAS A BOUND. It holds only while

        lesion diameter <= overlap in pixels = tile - stride

    Interior tiles start every `stride` px. A tile shows a lesion of diameter
    d whole iff the lesion centre is at least d/2 from both of its edges, i.e.
    in a window of width `tile - d`. Consecutive windows start `stride` apart,
    so they tile the axis without a hole iff stride <= tile - d, i.e.
    d <= tile - stride. Past that there are centres no tile shows whole, and
    `min_visible` starts deleting lesions that nothing else covers.

At the locked 640 px / 20% that overlap is 128 px, which at 26 px/mm is
4.9 mm. `max_safe_lesion_mm()` computes it, and the labeling tool must warn
when a lesion is drawn larger. Edge tiles are clamped, which only ever ADDS
overlap, so the interior case above is the worst case.

The one gap this does not cover is a lesion at the FRAME edge, whose disc
leaves the image entirely: no tile can show what is not in the photo. Pass
`image_shape` to `labels_in_tile` and visibility is measured against the
disc's in-image area instead, which is the question actually being asked -
"does this crop show as much of the lesion as exists" rather than "is the
lesion whole".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence

import numpy as np

TILE_PX = 640
OVERLAP = 0.2


@dataclass(frozen=True)
class Tile:
    """A half-open rect [x0, x1) x [y0, y1) in IMAGE pixel coordinates."""

    tile_id: str
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def shape(self) -> tuple[int, int]:
        """(h, w), matching numpy's convention."""
        return self.height, self.width

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def crop(self, img):
        """Slice this tile out of an HxW or HxWxC array."""
        return img[self.y0 : self.y1, self.x0 : self.x1]

    def to_dict(self) -> dict:
        return asdict(self)


MIN_VISIBLE = 0.5


# --------------------------------------------------------------------------
# disc / rect geometry
# --------------------------------------------------------------------------

def _quadrant_area(r: float, X: float, Y: float) -> float:
    """Area of {x^2 + y^2 <= r^2, 0 <= x <= X, 0 <= y <= Y}. X, Y >= 0.

    EXACT, not sampled. Sampling a disc on a small grid is the usual shortcut,
    but its error depends on the lesion's radius in pixels, so a fixed
    `min_visible` threshold would mean subtly different things for a 1.5 mm
    and a 5 mm lesion. This is closed form and costs a couple of arctangents.

        Q(X, Y) = integral_0^min(X,r) min(Y, sqrt(r^2 - x^2)) dx

    which splits at x_c = sqrt(r^2 - Y^2), the x where the circle drops below
    the height Y: flat at Y to the left of it, circular to the right.
    """
    if r <= 0 or X <= 0 or Y <= 0:
        return 0.0
    Xm = min(X, r)
    xc = np.sqrt(max(r * r - Y * Y, 0.0))
    flat = min(Xm, xc)
    area = Y * flat
    if Xm > flat:
        def anti(x: float) -> float:
            x = min(x, r)
            return 0.5 * (x * np.sqrt(max(r * r - x * x, 0.0)) + r * r * np.arcsin(x / r))
        area += anti(Xm) - anti(flat)
    return float(area)


def _disc_rect_area(cx: float, cy: float, r: float,
                    x0: float, y0: float, x1: float, y1: float) -> float:
    """Exact area of the intersection of disc((cx, cy), r) with a rect.

    The rect is shifted so the disc sits at the origin, split at the axes into
    up to four single-quadrant pieces, and each piece is mirrored into the
    positive quadrant where `_quadrant_area` applies. Within a quadrant the
    piece's area is the usual 2-D cumulative inclusion-exclusion.
    """
    if r <= 0 or x1 <= x0 or y1 <= y0:
        return 0.0
    a0, a1 = x0 - cx, x1 - cx
    b0, b1 = y0 - cy, y1 - cy

    total = 0.0
    for sx, ex, _ in _axis_pieces(a0, a1):
        for sy, ey, _ in _axis_pieces(b0, b1):
            total += (
                _quadrant_area(r, ex, ey)
                - _quadrant_area(r, sx, ey)
                - _quadrant_area(r, ex, sy)
                + _quadrant_area(r, sx, sy)
            )
    return total


def _axis_pieces(lo: float, hi: float) -> list[tuple[float, float, int]]:
    """Split an interval at 0 and mirror the negative part onto [0, inf).

    Returns (start, end, side) with 0 <= start <= end, so every piece can be
    handed to the positive-quadrant formula.
    """
    out: list[tuple[float, float, int]] = []
    if lo < 0:
        out.append((0.0, -lo if hi >= 0 else -hi, -1))
        if hi < 0:
            out[-1] = (-hi, -lo, -1)
    if hi > 0:
        out.append((lo if lo > 0 else 0.0, hi, +1))
    return [(s, e, k) for s, e, k in out if e > s]


def visible_fraction(
    cx: float, cy: float, r: float, tile: "Tile",
    image_shape: Sequence[int] | None = None,
) -> float:
    """Fraction of a lesion's disc that this tile shows. In [0, 1].

    A zero-radius label is treated as fully visible when its centre is in the
    tile: a point has no area to be clipped, and the alternative is dividing
    by zero over a rounding difference.

    With `image_shape` the denominator is the disc's IN-IMAGE area, so a
    lesion at the frame edge is not punished for the half of it that was never
    photographed. Without it the denominator is the whole disc.
    """
    if r <= 0:
        return 1.0 if tile.contains(cx, cy) else 0.0

    inside = _disc_rect_area(cx, cy, r, tile.x0, tile.y0, tile.x1, tile.y1)
    whole = np.pi * r * r
    if image_shape is not None:
        h, w = int(image_shape[0]), int(image_shape[1])
        whole = _disc_rect_area(cx, cy, r, 0, 0, w, h)
    if whole <= 0:
        return 0.0
    return float(min(1.0, inside / whole))


def max_safe_lesion_mm(px_per_mm: float, tile: int = TILE_PX,
                       overlap: float = OVERLAP) -> float:
    """Largest lesion DIAMETER, in mm, that the overlap can still show whole.

    Past this, `labels_in_tile`'s visibility filter can drop a lesion from
    every tile that contains its centre, because no tile shows enough of it -
    a silently missing training label. See the MEMBERSHIP section for the
    derivation; the bound is `tile - stride` pixels, which at 640 px / 20% and
    26 px/mm is 4.9 mm.

    The labeling tool must warn when a lesion is drawn larger than this. The
    answer at that point is not to lower `min_visible`; it is to raise the
    overlap (which re-tiles for free - labels are in image coordinates) or to
    accept that one lesion needs a whole-image view.
    """
    if px_per_mm <= 0:
        raise ValueError(f"px_per_mm must be positive, got {px_per_mm}")
    return (tile - stride_for(tile, overlap)) / float(px_per_mm)


def lesion_size_warning(diameter_mm: float, px_per_mm: float,
                        tile: int = TILE_PX, overlap: float = OVERLAP) -> str | None:
    """A warning string if this lesion is too big for the overlap, else None.

    For the labeling tool to call at draw time, while the annotation can still
    be reconsidered.
    """
    cap = max_safe_lesion_mm(px_per_mm, tile, overlap)
    if diameter_mm <= cap:
        return None
    return (
        f"lesion diameter {diameter_mm:.1f} mm exceeds {cap:.1f} mm, the largest "
        f"the {overlap:.0%} overlap on {tile} px tiles can show whole at "
        f"{px_per_mm:.1f} px/mm. No tile will contain all of it, so the "
        f"visibility filter may drop it from the training crops entirely. "
        f"Raise the overlap or label it from the full image."
    )


def stride_for(tile: int = TILE_PX, overlap: float = OVERLAP) -> int:
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    s = int(round(tile * (1.0 - overlap)))
    return max(1, s)


def _starts(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets along one axis. Last one clamped to the edge."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)      # clamp, do not pad
    return starts


def tiles_for(
    image_shape: Sequence[int], tile: int = TILE_PX, overlap: float = OVERLAP
) -> list[Tile]:
    """Deterministic tile grid for an image of this shape.

    `image_shape` is (h, w) or (h, w, c) - numpy's `.shape` works directly.

    Tile ids are derived from POSITION (`r{row}c{col}`), never from
    enumeration order or content, so the same shape always yields the same
    ids and a tile id remains meaningful across runs and across code changes.
    """
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    h, w = int(image_shape[0]), int(image_shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"image_shape must be positive, got {image_shape!r}")

    stride = stride_for(tile, overlap)
    ys = _starts(h, tile, stride)
    xs = _starts(w, tile, stride)

    out: list[Tile] = []
    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            out.append(
                Tile(
                    tile_id=f"r{r}c{c}",
                    row=r,
                    col=c,
                    x0=x0,
                    y0=y0,
                    x1=min(x0 + tile, w),
                    y1=min(y0 + tile, h),
                )
            )
    return out


def grid_shape(
    image_shape: Sequence[int], tile: int = TILE_PX, overlap: float = OVERLAP
) -> tuple[int, int]:
    """(n_rows, n_cols) without building the tiles."""
    h, w = int(image_shape[0]), int(image_shape[1])
    stride = stride_for(tile, overlap)
    return len(_starts(h, tile, stride)), len(_starts(w, tile, stride))


# --------------------------------------------------------------------------
# coordinate conversion
# --------------------------------------------------------------------------

def to_tile_coords(pt_image: Sequence[float], tile: Tile) -> tuple[float, float]:
    """IMAGE -> TILE. Pure integer translation.

    ROUND-TRIP GUARANTEE
      integers: exact. `to_image_coords(to_tile_coords(p, t), t) == p`.
                This is the form that gets persisted, so this is the case
                that has to be watertight, and it is.
      floats:   < 1e-9 px, NOT bit-exact.

    The guarantee is 1e-9 px. What actually happens is ~1e-12 px; those are
    two different claims and this docstring used to give them as one number.

    Bit-exact float round-tripping through a translation is impossible in
    IEEE-754: subtracting a large integer offset (x0 can be several thousand)
    discards low mantissa bits that adding it back cannot restore. Doubles
    carry ~15-16 significant digits, so at image coordinates below ~10000 the
    OBSERVED residual is ~1e-12 px. The GUARANTEED bound is 1e-9 px, three
    orders of magnitude of headroom above that, and it is what the test
    asserts and what `docs/DECISIONS.md` records.

    Either way it is far below the thing being measured: 1e-9 px at a
    realistic 26 px/mm is 4e-11 mm, against a 1.5 mm minimum lesion.

    The practical consequence is nil, because tile coordinates are transient
    UI state and are never persisted - see the COORDINATE CONTRACT above.
    """
    return (pt_image[0] - tile.x0, pt_image[1] - tile.y0)


def to_image_coords(pt_tile: Sequence[float], tile: Tile) -> tuple[float, float]:
    """TILE -> IMAGE. The only direction that may be persisted.

    See to_tile_coords for the round-trip guarantee.
    """
    return (pt_tile[0] + tile.x0, pt_tile[1] + tile.y0)


# --------------------------------------------------------------------------
# label membership
# --------------------------------------------------------------------------

def _xy(label: Any, x_key: str, y_key: str) -> tuple[float, float]:
    if isinstance(label, dict):
        return float(label[x_key]), float(label[y_key])
    return float(getattr(label, x_key)), float(getattr(label, y_key))


def _radius(label: Any, r_key: str) -> float:
    """Lesion radius in px, or 0.0 for a label that does not carry one.

    A missing radius means "point label", which is never dropped for
    visibility - there is no area to clip.
    """
    v = label.get(r_key) if isinstance(label, dict) else getattr(label, r_key, None)
    return float(v) if v is not None else 0.0


def labels_in_tile(
    labels: Iterable[Any],
    tile: Tile,
    x_key: str = "x_px",
    y_key: str = "y_px",
    r_key: str = "r_px",
    min_visible: float = MIN_VISIBLE,
    image_shape: Sequence[int] | None = None,
) -> list[Any]:
    """Every label centered in `tile` and at least `min_visible` visible in it.

    FOR TRAINING. A label near a seam is returned for every tile that contains
    its centre AND shows enough of it - that is intended. A tile crop that
    visibly shows a lesion must carry that lesion's label, otherwise the crop
    is a labelled false negative and actively teaches the detector to miss it.

    The visibility filter is the other half of that same argument. A lesion
    centered 5 px inside an edge is 90% outside the crop; keeping its label
    teaches the detector that a thin arc is a full-radius lesion. Dropping it
    is safe ONLY because a neighbouring tile shows it whole - which holds up
    to `max_safe_lesion_mm()` and no further. See the MEMBERSHIP section.

    `min_visible = 0.0` restores pure center-membership. A label with no
    radius (missing `r_key`, or zero) is treated as a point and is never
    dropped. Pass `image_shape` to measure against the disc's in-image area so
    that lesions at the frame edge are not punished for the part of them that
    was never photographed.

    Do NOT use this to count. Summing over tiles double-counts the overlap
    band; use assign_labels_to_tiles for that, which ignores visibility
    entirely - ownership is center-based, full stop.
    """
    if not 0.0 <= min_visible <= 1.0:
        raise ValueError(f"min_visible must be in [0, 1], got {min_visible}")

    out: list[Any] = []
    for l in labels:
        x, y = _xy(l, x_key, y_key)
        if not tile.contains(x, y):
            continue
        if min_visible <= 0.0:
            out.append(l)
            continue
        r = _radius(l, r_key)
        if r <= 0.0 or visible_fraction(x, y, r, tile, image_shape) >= min_visible:
            out.append(l)
    return out


def assign_labels_to_tiles(
    labels: Iterable[Any],
    tiles: Sequence[Tile],
    x_key: str = "x_px",
    y_key: str = "y_px",
) -> dict[str, list[Any]]:
    """Assign each label to exactly ONE tile. A strict partition.

    FOR COUNTING AND COVERAGE. Owner is the tile whose centre is nearest the
    lesion centre - which is the tile showing the most context around it -
    with ties broken by (row, col) for determinism.

    Ownership is CENTER-BASED ONLY. `labels_in_tile`'s visibility filter does
    not apply here and must not: a count that changed with how much of a
    lesion a crop happens to show would not be a count of lesions.

    Raises ValueError if any label falls outside every tile, rather than
    silently dropping it. A dropped label is a silent hole in ground truth.
    """
    out: dict[str, list[Any]] = {t.tile_id: [] for t in tiles}
    orphans: list[tuple[float, float]] = []

    for label in labels:
        x, y = _xy(label, x_key, y_key)
        best: Tile | None = None
        best_key: tuple[float, int, int] | None = None
        for t in tiles:
            if not t.contains(x, y):
                continue
            cx, cy = t.center
            key = ((cx - x) ** 2 + (cy - y) ** 2, t.row, t.col)
            if best_key is None or key < best_key:
                best, best_key = t, key
        if best is None:
            orphans.append((x, y))
        else:
            out[best.tile_id].append(label)

    if orphans:
        raise ValueError(
            f"{len(orphans)} label(s) fall outside every tile, e.g. {orphans[:3]}. "
            "Either the label is outside the image or the tile grid was built "
            "for a different image shape."
        )
    return out


def owning_tile(
    pt_image: Sequence[float], tiles: Sequence[Tile]
) -> Tile | None:
    """The single tile that owns this point, by the same rule as above."""
    x, y = float(pt_image[0]), float(pt_image[1])
    best: Tile | None = None
    best_key: tuple[float, int, int] | None = None
    for t in tiles:
        if not t.contains(x, y):
            continue
        cx, cy = t.center
        key = ((cx - x) ** 2 + (cy - y) ** 2, t.row, t.col)
        if best_key is None or key < best_key:
            best, best_key = t, key
    return best


# --------------------------------------------------------------------------
# invariants - importable so tests and callers check the same thing
# --------------------------------------------------------------------------

def covers_every_pixel(image_shape: Sequence[int], tiles: Sequence[Tile]) -> bool:
    """True if the union of tiles is the whole image.

    Checked by column/row interval union rather than a full boolean mask, so
    it is cheap enough to assert on a 4032x3024 frame.
    """
    h, w = int(image_shape[0]), int(image_shape[1])

    def _covers(intervals: list[tuple[int, int]], length: int) -> bool:
        reach = 0
        for a, b in sorted(intervals):
            if a > reach:
                return False
            reach = max(reach, b)
        return reach >= length

    return _covers([(t.x0, t.x1) for t in tiles], w) and _covers(
        [(t.y0, t.y1) for t in tiles], h
    )


def describe(image_shape: Sequence[int], tile: int = TILE_PX, overlap: float = OVERLAP) -> str:
    ts = tiles_for(image_shape, tile, overlap)
    rows, cols = grid_shape(image_shape, tile, overlap)
    h, w = int(image_shape[0]), int(image_shape[1])
    stride = stride_for(tile, overlap)
    # Clamping moves a tile's START back to the edge; it does not shrink the
    # tile unless the image is smaller than one tile. So "clamped" means
    # off-grid origin (extra overlap), and "undersized" means a small image.
    clamped = [t for t in ts if t.x0 % stride or t.y0 % stride]
    undersized = [t for t in ts if t.width < tile or t.height < tile]
    return (
        f"{w}x{h} -> {len(ts)} tiles ({rows}r x {cols}c) "
        f"@ {tile}px, {overlap:.0%} overlap, stride {stride}\n"
        f"  {len(clamped)} clamped (extra overlap at an edge), "
        f"{len(undersized)} undersized (image smaller than a tile); "
        f"covers every pixel: {covers_every_pixel(image_shape, ts)}"
    )


if __name__ == "__main__":
    for shape in [(3024, 4032), (4032, 3024), (640, 640), (500, 300), (1000, 640)]:
        print(describe(shape))
        print()
    for ppm in (20.0, 26.0, 34.0):
        print(f"at {ppm:.0f} px/mm, overlap shows a lesion whole up to "
              f"{max_safe_lesion_mm(ppm):.1f} mm across")
