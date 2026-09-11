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

    labels_in_tile()         every label centered in this tile.
                             A label may appear in several tiles.
                             USE FOR TRAINING - a tile that renders a lesion
                             must carry its label, or the crop teaches the
                             detector that a visible lesion is background.

    assign_labels_to_tiles() exactly one owning tile per label.
                             USE FOR COUNTING AND COVERAGE.
                             Owner = tile whose centre is nearest the lesion
                             (most surrounding context), ties broken by
                             (row, col) so it is deterministic.

This mirrors `detections.owned` for the pose overlap band: kept everywhere it
is visible, counted in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence

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
      floats:   exact to within ~1e-9 px, NOT bit-exact.

    Bit-exact float round-tripping through a translation is impossible in
    IEEE-754: subtracting a large integer offset (x0 can be several thousand)
    discards low mantissa bits that adding it back cannot restore. Doubles
    carry ~15-16 significant digits, so at image coordinates below ~10000 the
    residual is ~1e-12 px, which at a realistic 26 px/mm is 4e-14 mm. It is
    many orders of magnitude below the thing being measured; the rubric's
    minimum lesion is 1.5 mm.

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


def labels_in_tile(
    labels: Iterable[Any], tile: Tile, x_key: str = "x_px", y_key: str = "y_px"
) -> list[Any]:
    """Every label whose CENTER falls inside `tile`.

    FOR TRAINING. A label near a seam is returned for every tile that
    contains its centre - that is intended. A tile crop that visibly shows a
    lesion must carry that lesion's label, otherwise the crop is a labelled
    false negative and actively teaches the detector to miss it.

    Do NOT use this to count. Summing over tiles double-counts the overlap
    band; use assign_labels_to_tiles for that.
    """
    return [l for l in labels if tile.contains(*_xy(l, x_key, y_key))]


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
