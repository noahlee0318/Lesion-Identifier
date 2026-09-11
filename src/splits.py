"""Date-based train/val/test splits. Small file, large consequence.

Consecutive days in this dataset are near-duplicates: the same face, the same
rig, the same lamp, twenty-four hours apart. A random split puts Tuesday in
train and Wednesday in test and hands back excellent metrics that mean
nothing - and that failure is invisible from the outside. The only defence is
to make the split structurally incapable of producing it.

THREE PROPERTIES, ENFORCED RATHER THAN INTENDED

1. Contiguous blocks, in time order: train | gap | val | gap | test.
   Not interleaved. Interleaving still leaks, because day N and day N+1 are
   nearly the same image regardless of which side of the split they land on.

2. Earliest dates train, latest dates test. Train on the past, test on the
   future. That is the only arrangement matching how the system is used.

3. A calendar washout gap between blocks. Test starting the day after train
   ends still puts near-duplicate days on both sides. Gap days are excluded
   from every split, not reassigned.

NO RANDOMNESS. There is no `shuffle` argument, no `random_state`, no code
path in this module that touches an RNG - `test_splits.py` asserts that at
the source level. Wanting a random split should require writing a new
function and confronting the decision in a diff.

WHY CALENDAR DAYS, NOT DATE-INDEX

The gap is measured in real calendar days between block boundaries, not in
"however many sessions happen to be next in the list". If a week is missed,
the dates either side of the hole are already far apart in time and no
washout is needed; if capture is dense, three sessions is three days. Only
the calendar answers "are these two images near-duplicates".

CALIBRATION IMAGES ARE NEVER IN A SPLIT. They are throwaway shots taken
before the rig was locked, at one pose, on two lenses. `sessions.kind` is
filtered to 'session' throughout.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

try:
    from . import config
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_GAP_DAYS = 3


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


class NotEnoughDataError(RuntimeError):
    """Raised when there are too few dates to split honestly."""


@dataclass
class DatePlan:
    """Which dates land in which block. Pure function of the inputs."""

    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    gap_days: int = DEFAULT_GAP_DAYS
    ratios: tuple[float, float, float] = DEFAULT_RATIOS

    def as_dict(self) -> dict:
        return asdict(self)


def plan_split(
    dates: Sequence[str],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    gap_days: int = DEFAULT_GAP_DAYS,
) -> DatePlan:
    """Chronological blocks with a calendar washout between them.

    Boundaries are chosen by proportion of available dates, then the washout
    is applied by DROPPING dates from the START of the later block. Dropping
    from the later block rather than the earlier one keeps the maximum amount
    of training data and preserves "test is the most recent data".

    Raises NotEnoughDataError, loudly and with numbers, rather than returning
    a degenerate split. For the first couple of weeks of capture this is the
    expected outcome and the message says so.
    """
    if gap_days < 0:
        raise ValueError(f"gap_days must be >= 0, got {gap_days}")
    if len(ratios) != 3 or any(r < 0 for r in ratios) or sum(ratios) <= 0:
        raise ValueError(f"ratios must be three non-negative numbers, got {ratios!r}")

    uniq = sorted(set(dates))
    for s in uniq:
        _d(s)                                   # validate format early
    n = len(uniq)
    if n == 0:
        raise NotEnoughDataError(
            "no session dates at all. Nothing to split - keep shooting."
        )

    total = float(sum(ratios))
    r = tuple(x / total for x in ratios)

    # Provisional boundaries over all dates, before the washout.
    n_train = max(1, int(round(n * r[0])))
    n_val = max(1, int(round(n * r[1])))
    if n_train + n_val >= n:                    # leave at least one for test
        n_train = max(1, n - 2)
        n_val = max(1, n - n_train - 1)

    train = uniq[:n_train]
    val_c = uniq[n_train : n_train + n_val]
    test_c = uniq[n_train + n_val :]

    # Calendar washout: drop from the front of each later block.
    excluded: list[str] = []

    def _washout(prev_block: list[str], cand: list[str]) -> list[str]:
        if not prev_block or not cand or gap_days == 0:
            return cand
        last = _d(prev_block[-1])
        keep = [s for s in cand if (_d(s) - last).days > gap_days]
        excluded.extend([s for s in cand if s not in keep])
        return keep

    val = _washout(train, val_c)
    test = _washout(val if val else train, test_c)

    empty = [nm for nm, blk in zip(SPLIT_NAMES, (train, val, test)) if not blk]
    if empty:
        span = f"{uniq[0]} .. {uniq[-1]}"
        need = 1 + (gap_days + 1) + (gap_days + 1)
        raise NotEnoughDataError(
            f"cannot build a {ratios} split with a {gap_days}-day washout: "
            f"{', '.join(empty)} would be empty.\n"
            f"  have {n} session date(s) ({span})\n"
            f"  need roughly {need}+ dates at this gap for a minimal split, "
            f"and many more for the ratios to mean anything.\n"
            f"  This is the expected state for the first couple of weeks of "
            f"capture. Keep shooting; do not lower the gap to force it."
        )

    return DatePlan(
        train=train,
        val=val,
        test=test,
        excluded=sorted(excluded),
        gap_days=gap_days,
        ratios=tuple(ratios),
    )


# --------------------------------------------------------------------------
# assertions - properties, checked rather than assumed
# --------------------------------------------------------------------------

def check_plan(plan: DatePlan) -> None:
    """Verify every property this module promises. Raises AssertionError."""
    blocks = {"train": plan.train, "val": plan.val, "test": plan.test}

    for name, blk in blocks.items():
        assert blk, f"{name} split is empty"
        assert blk == sorted(blk), f"{name} dates are not sorted"
        assert len(set(blk)) == len(blk), f"{name} contains a duplicate date"

    # No date in more than one split, and none of them is an excluded date.
    seen: dict[str, str] = {}
    for name, blk in blocks.items():
        for s in blk:
            assert s not in seen, f"{s} is in both {seen[s]} and {name}"
            seen[s] = name
    for s in plan.excluded:
        assert s not in seen, f"washout date {s} also appears in {seen[s]}"

    # Chronological, non-overlapping blocks.
    assert _d(plan.train[-1]) < _d(plan.val[0]), "train does not precede val"
    assert _d(plan.val[-1]) < _d(plan.test[0]), "val does not precede test"

    # The washout is actually respected, in calendar days.
    if plan.gap_days > 0:
        g1 = (_d(plan.val[0]) - _d(plan.train[-1])).days
        g2 = (_d(plan.test[0]) - _d(plan.val[-1])).days
        assert g1 > plan.gap_days, (
            f"only {g1} day(s) between train and val, need > {plan.gap_days}"
        )
        assert g2 > plan.gap_days, (
            f"only {g2} day(s) between val and test, need > {plan.gap_days}"
        )


# --------------------------------------------------------------------------
# database-backed split
# --------------------------------------------------------------------------

@dataclass
class Split:
    name: str
    created_at: str
    ratios: tuple[float, float, float]
    gap_days: int
    dates: dict[str, list[str]]
    images: dict[str, list[int]]
    sessions: dict[str, list[int]]
    excluded_dates: list[str]
    counts: dict[str, dict[str, int]]

    def to_dict(self) -> dict:
        return asdict(self)


def _session_dates(con: sqlite3.Connection) -> list[str]:
    rows = con.execute(
        "SELECT DISTINCT session_date FROM sessions WHERE kind='session' "
        "ORDER BY session_date"
    ).fetchall()
    return [r[0] for r in rows]


def _images_for(con: sqlite3.Connection, dates: Sequence[str]) -> tuple[list[int], list[int]]:
    if not dates:
        return [], []
    q = ",".join("?" * len(dates))
    rows = con.execute(
        f"SELECT i.id, s.id FROM images i JOIN sessions s ON s.id=i.session_id "
        f"WHERE s.kind='session' AND s.session_date IN ({q}) ORDER BY i.id",
        tuple(dates),
    ).fetchall()
    return [r[0] for r in rows], sorted({r[1] for r in rows})


def build_split(
    con: sqlite3.Connection,
    name: str = "v1",
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    gap_days: int = DEFAULT_GAP_DAYS,
    persist: bool = True,
) -> Split:
    """Resolve a split against the database and (by default) write it down.

    Persisting is not optional bookkeeping: a metric computed against an
    unrecorded split is not reproducible, and six months from now "0.82
    precision" with no record of which days were held out is worthless.
    """
    plan = plan_split(_session_dates(con), ratios, gap_days)
    check_plan(plan)

    dates = {"train": plan.train, "val": plan.val, "test": plan.test}
    images: dict[str, list[int]] = {}
    sessions: dict[str, list[int]] = {}
    for k, ds in dates.items():
        images[k], sessions[k] = _images_for(con, ds)

    # No image may appear twice. Dates are disjoint so this should be
    # impossible, but an image mis-joined to two sessions would slip past the
    # date check and leak silently - which is the exact failure mode this
    # module exists to prevent.
    seen: dict[int, str] = {}
    for k, ids in images.items():
        for i in ids:
            assert i not in seen, f"image {i} is in both {seen[i]} and {k}"
            seen[i] = k

    counts = {
        k: {
            "days": len(dates[k]),
            "sessions": len(sessions[k]),
            "images": len(images[k]),
        }
        for k in SPLIT_NAMES
    }

    split = Split(
        name=name,
        created_at=datetime.now().isoformat(timespec="seconds"),
        ratios=tuple(ratios),
        gap_days=gap_days,
        dates=dates,
        images=images,
        sessions=sessions,
        excluded_dates=plan.excluded,
        counts=counts,
    )
    if persist:
        save_split(split)
    return split


def splits_dir() -> Path:
    d = config.DERIVED_DIR / "splits"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_split(split: Split) -> Path:
    p = splits_dir() / f"{split.name}.json"
    p.write_text(json.dumps(split.to_dict(), indent=2), encoding="utf-8")
    return p


def load_split(name: str) -> Split:
    p = splits_dir() / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no split named {name!r} at {p}. Build it with:  "
            f"python -m src.splits --name {name}"
        )
    d = json.loads(p.read_text(encoding="utf-8"))
    d["ratios"] = tuple(d["ratios"])
    return Split(**d)


def format_report(split: Split) -> str:
    L = [
        f"split '{split.name}'   created {split.created_at}",
        f"ratios {split.ratios}   washout {split.gap_days} calendar day(s)",
        "",
        f"{'':<6} {'days':>5} {'sessions':>9} {'images':>7}   date range",
        "-" * 62,
    ]
    for k in SPLIT_NAMES:
        c = split.counts[k]
        ds = split.dates[k]
        rng = f"{ds[0]} .. {ds[-1]}" if ds else "-"
        L.append(f"{k:<6} {c['days']:>5} {c['sessions']:>9} {c['images']:>7}   {rng}")
    L.append("-" * 62)
    tot = {k: sum(split.counts[s][k] for s in SPLIT_NAMES) for k in ("days", "sessions", "images")}
    L.append(f"{'total':<6} {tot['days']:>5} {tot['sessions']:>9} {tot['images']:>7}")
    if split.excluded_dates:
        L.append("")
        L.append(f"washout, excluded from every split ({len(split.excluded_dates)} day(s)):")
        L.append("  " + ", ".join(split.excluded_dates))
    L.append("")
    L.append("Train is the past, test is the future. Gap days belong to no split.")
    return "\n".join(L)


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="date-based train/val/test split")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--ratios", type=float, nargs=3, default=list(DEFAULT_RATIOS),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--gap", type=int, default=DEFAULT_GAP_DAYS,
                    help="calendar washout days between blocks")
    ap.add_argument("--show", action="store_true", help="load and print an existing split")
    ap.add_argument("--dry-run", action="store_true", help="do not write the json")
    a = ap.parse_args(argv)

    if a.show:
        print(format_report(load_split(a.name)))
        return 0

    from src.server import db as _db

    con = _db.connect()
    try:
        split = build_split(
            con, a.name, tuple(a.ratios), a.gap, persist=not a.dry_run
        )
    except NotEnoughDataError as e:
        print("NOT ENOUGH DATA\n")
        print(e)
        return 2
    print(format_report(split))
    if not a.dry_run:
        print(f"\nwrote {splits_dir() / (a.name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
