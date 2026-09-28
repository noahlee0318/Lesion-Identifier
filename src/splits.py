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

WHERE THE BOUNDARIES GO: FIXED TRAILING WINDOWS, NOT RATIOS

The default is `mode="window"`: the last `test_days` (14) are test, the
`val_days` (14) before that are val, everything earlier is train, with the
calendar washout between blocks.

Proportional boundaries were the original rule and are still available as
`mode="ratio"`, but they are the wrong default here, because the held-out set
is supposed to answer one fixed question - "how does the detector do on
recent, unseen days" - and a proportional test set is a different size every
time it is built. Early on it is three days, which is too thin to measure
anything and swings wildly with one bad session. A year in it is fifty-five
days, which is fifty days more than the question needs and fifty days stolen
from training. A fixed window is the same size forever, always current, and
every rebuild is comparable to the last.

The cost, stated plainly: for the first weeks train is SMALLER than test.
That is not a flaw in the rule, it is an honest report that there is not much
data yet, and NotEnoughDataError covers the part where it would be degenerate.

Under the fixed-window rule a split needs roughly

    14 (test) + 3 (gap) + 14 (val) + 3 (gap) + 1 (train) = 35 calendar days

of capture before it can be built at all. Before that, plan_split raises.

CALIBRATION IMAGES ARE NEVER IN A SPLIT. They are throwaway shots taken
before the rig was locked, at one pose, on two lenses. `sessions.kind` is
filtered to 'session' throughout.

SPLIT FILES ARE WRITE-ONCE. A metric is meaningless without the split it was
computed against, so `save_split` refuses to overwrite an existing split file
and the filename carries a content hash. See save_split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
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

# Fixed trailing windows, in calendar days. Both blocks the same size: they
# answer the same kind of question and there is no reason for val to be
# smaller than the thing it is a rehearsal for.
DEFAULT_TEST_DAYS = 14
DEFAULT_VAL_DAYS = 14

MODES = ("window", "ratio")


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def min_days_for_window(
    gap_days: int = DEFAULT_GAP_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
    val_days: int = DEFAULT_VAL_DAYS,
) -> int:
    """Calendar span a fixed-window split needs before it can exist at all."""
    return test_days + gap_days + val_days + gap_days + 1


class NotEnoughDataError(RuntimeError):
    """Raised when there are too few dates to split honestly."""


class SplitExistsError(FileExistsError):
    """Raised when writing a split would destroy an existing one."""


def _ratios_of(train: Sequence[str], val: Sequence[str], test: Sequence[str]):
    """Achieved ratios, over the days that landed in a block.

    Washout days are excluded from the denominator so the three numbers sum to
    one and can be read against the ratios that were requested.
    """
    n = len(train) + len(val) + len(test)
    if n == 0:
        return (0.0, 0.0, 0.0)
    return tuple(round(len(b) / n, 4) for b in (train, val, test))


@dataclass
class DatePlan:
    """Which dates land in which block. Pure function of the inputs."""

    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    gap_days: int = DEFAULT_GAP_DAYS
    mode: str = "window"
    # Requested proportions. None under the fixed-window rule, which does not
    # have any - recording 0.70/0.15/0.15 there would be a claim about the
    # split that is not true of it.
    ratios: tuple[float, float, float] | None = None
    test_days: int | None = DEFAULT_TEST_DAYS
    val_days: int | None = DEFAULT_VAL_DAYS
    achieved_ratios: tuple[float, float, float] = field(init=False, default=(0.0, 0.0, 0.0))

    def __post_init__(self) -> None:
        self.achieved_ratios = _ratios_of(self.train, self.val, self.test)

    def as_dict(self) -> dict:
        return asdict(self)


def _too_few(uniq: list[str], why: str, need: str) -> NotEnoughDataError:
    span = f"{uniq[0]} .. {uniq[-1]}" if uniq else "-"
    n = len(uniq)
    return NotEnoughDataError(
        f"{why}\n"
        f"  have {n} session date(s) ({span})\n"
        f"  {need}\n"
        f"  This is the expected state for the first few weeks of capture. "
        f"Keep shooting; do not shrink the windows or the gap to force it."
    )


def _plan_window(
    uniq: list[str], gap_days: int, test_days: int, val_days: int
) -> DatePlan:
    """Fixed trailing calendar windows. The default rule - see the docstring."""
    last = _d(uniq[-1])

    # Boundaries are computed from the CALENDAR, not from how many sessions
    # happen to fall in each window. A sparse fortnight gives a small test set,
    # which is the honest answer: there were few sessions in the last 14 days.
    test_start = last - timedelta(days=test_days - 1)
    val_end = test_start - timedelta(days=gap_days + 1)
    val_start = val_end - timedelta(days=val_days - 1)
    train_end = val_start - timedelta(days=gap_days + 1)

    train = [s for s in uniq if _d(s) <= train_end]
    val = [s for s in uniq if val_start <= _d(s) <= val_end]
    test = [s for s in uniq if _d(s) >= test_start]
    inside = set(train) | set(val) | set(test)
    excluded = sorted(s for s in uniq if s not in inside)

    empty = [nm for nm, blk in zip(SPLIT_NAMES, (train, val, test)) if not blk]
    if empty:
        need_days = min_days_for_window(gap_days, test_days, val_days)
        span_days = (last - _d(uniq[0])).days + 1
        raise _too_few(
            uniq,
            f"cannot build a fixed-window split ({test_days}d test, {val_days}d val, "
            f"{gap_days}-day washout): {', '.join(empty)} would be empty.",
            f"that rule needs about {need_days} calendar days of capture "
            f"({test_days} + {gap_days} + {val_days} + {gap_days} + 1); "
            f"the data spans {span_days}.",
        )
    return DatePlan(
        train=train, val=val, test=test, excluded=excluded, gap_days=gap_days,
        mode="window", ratios=None, test_days=test_days, val_days=val_days,
    )


def _plan_ratio(
    uniq: list[str], gap_days: int, ratios: tuple[float, float, float]
) -> DatePlan:
    """Proportional boundaries, then washout. The explicitly-named alternative.

    Kept because there are legitimate uses - a one-off retrospective split over
    a finished capture period, where "the last 14 days" is not the question
    being asked. It is not the default; see the module docstring.
    """
    n = len(uniq)
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
        need = 1 + (gap_days + 1) + (gap_days + 1)
        raise _too_few(
            uniq,
            f"cannot build a {ratios} split with a {gap_days}-day washout: "
            f"{', '.join(empty)} would be empty.",
            f"need roughly {need}+ dates at this gap for a minimal split, "
            f"and many more for the ratios to mean anything.",
        )
    return DatePlan(
        train=train, val=val, test=test, excluded=sorted(excluded),
        gap_days=gap_days, mode="ratio", ratios=tuple(ratios),
        test_days=None, val_days=None,
    )


def plan_split(
    dates: Sequence[str],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    gap_days: int = DEFAULT_GAP_DAYS,
    mode: str = "window",
    test_days: int = DEFAULT_TEST_DAYS,
    val_days: int = DEFAULT_VAL_DAYS,
) -> DatePlan:
    """Chronological blocks with a calendar washout between them.

    `mode="window"` (default) puts the last `test_days` in test and the
    `val_days` before that in val - a fixed-size, always-current held-out set.
    `mode="ratio"` uses `ratios` instead, which is the older proportional rule;
    see the module docstring for why it is no longer the default.

    `ratios` is validated whichever mode is in force, so a caller that passes
    nonsense hears about it rather than having it silently ignored.

    Raises NotEnoughDataError, loudly and with numbers, rather than returning
    a degenerate split. For the first weeks of capture this is the expected
    outcome and the message says so.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if gap_days < 0:
        raise ValueError(f"gap_days must be >= 0, got {gap_days}")
    if len(ratios) != 3 or any(r < 0 for r in ratios) or sum(ratios) <= 0:
        raise ValueError(f"ratios must be three non-negative numbers, got {ratios!r}")
    if test_days < 1 or val_days < 1:
        raise ValueError(
            f"test_days and val_days must be >= 1, got {test_days} and {val_days}"
        )

    uniq = sorted(set(dates))
    for s in uniq:
        _d(s)                                   # validate format early
    if not uniq:
        raise NotEnoughDataError(
            "no session dates at all. Nothing to split - keep shooting."
        )

    if mode == "window":
        return _plan_window(uniq, gap_days, test_days, val_days)
    return _plan_ratio(uniq, gap_days, ratios)


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
    gap_days: int
    dates: dict[str, list[str]]
    images: dict[str, list[int]]
    sessions: dict[str, list[int]]
    excluded_dates: list[str]
    counts: dict[str, dict[str, int]]
    mode: str = "window"
    # What was ASKED for (ratio mode only) and what was GOT. They differ: a
    # requested 0.70/0.15/0.15 lands near 0.78/0.11/0.11 once a 3-day washout
    # has eaten the front of val and test. Recording only the request makes the
    # file misleading to read six months from now.
    ratios: tuple[float, float, float] | None = None
    achieved_ratios: tuple[float, float, float] = (0.0, 0.0, 0.0)
    test_days: int | None = None
    val_days: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def content_hash(self) -> str:
        """Short hash of everything that defines the split.

        `name` and `created_at` are excluded, so rebuilding an unchanged
        dataset reproduces the same hash and is recognisable as the same
        split rather than as a collision.
        """
        return content_hash_of(self)


def content_hash_of(split: "Split") -> str:
    payload = split.to_dict()
    payload.pop("name", None)
    payload.pop("created_at", None)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


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
    mode: str = "window",
    test_days: int = DEFAULT_TEST_DAYS,
    val_days: int = DEFAULT_VAL_DAYS,
    force: bool = False,
) -> Split:
    """Resolve a split against the database and (by default) write it down.

    Persisting is not optional bookkeeping: a metric computed against an
    unrecorded split is not reproducible, and six months from now "0.82
    precision" with no record of which days were held out is worthless.

    Raises SplitExistsError rather than overwriting a split of the same name -
    see save_split.
    """
    plan = plan_split(
        _session_dates(con), ratios, gap_days,
        mode=mode, test_days=test_days, val_days=val_days,
    )
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
        gap_days=gap_days,
        dates=dates,
        images=images,
        sessions=sessions,
        excluded_dates=plan.excluded,
        counts=counts,
        mode=plan.mode,
        ratios=plan.ratios,
        achieved_ratios=plan.achieved_ratios,
        test_days=plan.test_days,
        val_days=plan.val_days,
    )
    if persist:
        save_split(split, force=force)
    return split


def splits_dir() -> Path:
    d = config.DERIVED_DIR / "splits"
    d.mkdir(parents=True, exist_ok=True)
    return d


def split_path(split: Split) -> Path:
    """Where this split belongs: `{name}.{content hash}.json`.

    The hash in the FILENAME is the point. A name alone can silently mean two
    different things - "the v1 split" in a metrics log six months from now is
    only a reference if exactly one file was ever called that. With the hash,
    a rebuild that changed anything lands on a different filename and cannot
    be mistaken for the original, and a rebuild that changed nothing lands on
    the same one and is recognisably the same split.
    """
    return splits_dir() / f"{split.name}.{content_hash_of(split)}.json"


def existing_splits(name: str) -> list[Path]:
    """Every file already claiming this split name."""
    return sorted(splits_dir().glob(f"{name}.*.json"))


def _identify(p: Path) -> str:
    """One line describing a split file, for a message about destroying it."""
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        c = d.get("counts", {})
        dates = d.get("dates", {})
        parts = []
        for k in SPLIT_NAMES:
            ds = dates.get(k) or []
            rng = f"{ds[0]}..{ds[-1]}" if ds else "-"
            parts.append(f"{k} {c.get(k, {}).get('images', '?')} img over "
                         f"{c.get(k, {}).get('days', '?')}d ({rng})")
        return f"{p.name}  created {d.get('created_at', '?')}  " + "; ".join(parts)
    except Exception:  # noqa: BLE001
        return f"{p.name}  (unreadable: not valid split JSON)"


def save_split(split: Split, force: bool = False) -> Path:
    """Write a split down. REFUSES to overwrite one that already exists.

    This module exists to make metrics reproducible, and a silent overwrite
    defeats it with its own default: rebuild `v1` next month with more data
    and the split last month's numbers were computed against is simply gone,
    with nothing in the output to say so.

    So: if any file already claims this name, raise SplitExistsError naming
    what is there and when it was made. Pick a new name - `v2`, or
    `v1-2026-10` - rather than reusing one.

    `force=True` is the deliberate exception. It prints what it is about to
    destroy before destroying it.

    Rebuilding an UNCHANGED dataset is not a conflict: the content hash is the
    same, so it is the same split, and the original file (with its original
    created_at) is kept and returned.
    """
    p = split_path(split)
    mine = content_hash_of(split)

    if p.exists():
        try:
            on_disk = json.loads(p.read_text(encoding="utf-8"))
            same = _hash_of_dict(on_disk) == mine
        except Exception:  # noqa: BLE001
            same = False
        if same:
            # Byte-identical content under a different timestamp. Keep the
            # original created_at - it is the date this split first existed,
            # which is the useful fact - and adopt it onto the in-memory object
            # so a report printed from it matches the file it refers to.
            split.created_at = on_disk.get("created_at", split.created_at)
            return p

    clashes = [q for q in existing_splits(split.name) if q != p or p.exists()]
    if clashes and not force:
        listing = "\n".join("    " + _identify(q) for q in clashes)
        raise SplitExistsError(
            f"a split named {split.name!r} already exists and would be replaced:\n"
            f"{listing}\n"
            f"  new split would be: {p.name}  "
            f"({', '.join(f'{k} {split.counts[k]['images']} img' for k in SPLIT_NAMES)})\n"
            f"\n"
            f"  Metrics already computed against the existing split refer to THOSE days. "
            f"Overwriting it makes them unreproducible and there would be nothing left to "
            f"say so.\n"
            f"  Pick a new name (--name v2, --name {split.name}-"
            f"{datetime.now():%Y-%m}), or pass --force if you really mean to destroy it."
        )
    if clashes and force:
        print("--force: DESTROYING these split files:")
        for q in clashes:
            print("    " + _identify(q))
            q.unlink()
        print(f"  replacing with {p.name}")

    p.write_text(json.dumps(split.to_dict(), indent=2), encoding="utf-8")
    return p


def _hash_of_dict(d: dict) -> str:
    payload = dict(d)
    payload.pop("name", None)
    payload.pop("created_at", None)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


class SplitIntegrityError(ValueError):
    """A split file on disk does not satisfy the invariants it was built with."""


def validate_split(split: Split) -> None:
    """Re-run every invariant against a split that came from disk.

    `load_split` used to parse JSON and hand it back unchecked. A hand-edited
    or truncated file would then produce metrics that look normal and are
    wrong - the single failure this module exists to prevent, arriving through
    the back door. Checking on load costs microseconds.
    """
    for k in SPLIT_NAMES:
        if k not in split.dates or k not in split.images or k not in split.counts:
            raise SplitIntegrityError(f"split {split.name!r} has no {k!r} block")

    plan = DatePlan(
        train=list(split.dates["train"]),
        val=list(split.dates["val"]),
        test=list(split.dates["test"]),
        excluded=list(split.excluded_dates),
        gap_days=split.gap_days,
        mode=split.mode,
        ratios=split.ratios,
        test_days=split.test_days,
        val_days=split.val_days,
    )
    try:
        check_plan(plan)
    except AssertionError as e:
        raise SplitIntegrityError(
            f"split {split.name!r} violates a split invariant: {e}\n"
            "  It was either hand-edited or written by a different version of "
            "this module. Do not compute metrics against it - rebuild it."
        ) from e

    # Images and sessions must be disjoint across blocks.
    for field_name in ("images", "sessions"):
        table = getattr(split, field_name)
        seen: dict[int, str] = {}
        for k in SPLIT_NAMES:
            for i in table[k]:
                if i in seen:
                    raise SplitIntegrityError(
                        f"{field_name[:-1]} {i} is in both {seen[i]} and {k} "
                        f"in split {split.name!r}"
                    )
                seen[i] = k

    # Counts must describe the lists they claim to describe.
    for k in SPLIT_NAMES:
        for key, seq in (("days", split.dates[k]), ("images", split.images[k]),
                         ("sessions", split.sessions[k])):
            if split.counts[k].get(key) != len(seq):
                raise SplitIntegrityError(
                    f"split {split.name!r}: counts[{k}][{key}] says "
                    f"{split.counts[k].get(key)} but the list has {len(seq)}"
                )

    recomputed = _ratios_of(split.dates["train"], split.dates["val"], split.dates["test"])
    if tuple(split.achieved_ratios) != recomputed:
        raise SplitIntegrityError(
            f"split {split.name!r}: achieved_ratios says {tuple(split.achieved_ratios)} "
            f"but the day counts give {recomputed}"
        )


def load_split(name: str) -> Split:
    """Load a split by name or by full stem, and VALIDATE it before returning.

    `name` may be the plain name (`v1`) or the full stem including its content
    hash (`v1.a3f81c2d`). A plain name that matches several files is an error
    rather than a guess - that ambiguity is exactly what the hash exists to
    surface.
    """
    direct = splits_dir() / f"{name}.json"
    matches = existing_splits(name)
    if direct.exists():
        p = direct
    elif len(matches) == 1:
        p = matches[0]
    elif not matches:
        raise FileNotFoundError(
            f"no split named {name!r} in {splits_dir()}. Build it with:  "
            f"python -m src.splits --name {name}"
        )
    else:
        listing = "\n".join("    " + _identify(q) for q in matches)
        raise ValueError(
            f"{len(matches)} files claim the split name {name!r}:\n{listing}\n"
            f"  Load one by its full stem, e.g. "
            f"load_split({matches[0].stem!r})."
        )

    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("ratios") is not None:
        d["ratios"] = tuple(d["ratios"])
    d["achieved_ratios"] = tuple(d.get("achieved_ratios", (0.0, 0.0, 0.0)))
    try:
        split = Split(**d)
    except TypeError as e:
        raise SplitIntegrityError(
            f"{p.name} is not a split file this version can read: {e}"
        ) from e
    validate_split(split)
    return split


def format_report(split: Split) -> str:
    if split.mode == "window":
        rule = (f"fixed trailing windows: {split.test_days}d test, "
                f"{split.val_days}d val")
    else:
        rule = f"requested ratios {split.ratios}"
    L = [
        f"split '{split.name}'   created {split.created_at}   "
        f"[{content_hash_of(split)}]",
        f"{rule}   washout {split.gap_days} calendar day(s)",
        f"achieved {split.achieved_ratios[0]:.2f} / {split.achieved_ratios[1]:.2f} "
        f"/ {split.achieved_ratios[2]:.2f}  (by day count, washout days excluded)",
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
    if split.mode == "window" and split.counts["train"]["days"] < split.counts["test"]["days"]:
        L.append(
            f"NOTE: train ({split.counts['train']['days']}d) is smaller than test "
            f"({split.counts['test']['days']}d). That is the fixed-window rule "
            f"reporting honestly that there is not much data yet, not a bug."
        )
    return "\n".join(L)


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="date-based train/val/test split")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--mode", choices=MODES, default="window",
                    help="window: fixed trailing calendar windows (default). "
                         "ratio: proportional boundaries.")
    ap.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS,
                    help="window mode: calendar days in the test block")
    ap.add_argument("--val-days", type=int, default=DEFAULT_VAL_DAYS,
                    help="window mode: calendar days in the val block")
    ap.add_argument("--ratios", type=float, nargs=3, default=list(DEFAULT_RATIOS),
                    metavar=("TRAIN", "VAL", "TEST"), help="ratio mode only")
    ap.add_argument("--gap", type=int, default=DEFAULT_GAP_DAYS,
                    help="calendar washout days between blocks")
    ap.add_argument("--show", action="store_true", help="load and print an existing split")
    ap.add_argument("--dry-run", action="store_true", help="do not write the json")
    ap.add_argument("--force", action="store_true",
                    help="destroy an existing split of this name (prints what it destroys)")
    a = ap.parse_args(argv)

    if a.show:
        print(format_report(load_split(a.name)))
        return 0

    from src.server import db as _db

    con = _db.connect()
    try:
        split = build_split(
            con, a.name, tuple(a.ratios), a.gap, persist=not a.dry_run,
            mode=a.mode, test_days=a.test_days, val_days=a.val_days, force=a.force,
        )
    except NotEnoughDataError as e:
        print("NOT ENOUGH DATA\n")
        print(e)
        return 2
    except SplitExistsError as e:
        print("REFUSING TO OVERWRITE\n")
        print(e)
        return 3
    print(format_report(split))
    if not a.dry_run:
        print(f"\nwrote {split_path(split)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
