"""Tests for the date-based split.

The point of these is not that the arithmetic works - it is that the split is
structurally incapable of leaking. Leakage here produces excellent metrics
that mean nothing, and nothing else in the pipeline would catch it.

Runs standalone (no pytest needed):
    python -m tests.test_splits
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.splits import (  # noqa: E402
    DEFAULT_GAP_DAYS,
    NotEnoughDataError,
    build_split,
    check_plan,
    plan_split,
)


def days(n: int, start: str = "2026-01-01", skip: set[int] | None = None) -> list[str]:
    d0 = date.fromisoformat(start)
    skip = skip or set()
    return [
        (d0 + timedelta(days=i)).isoformat() for i in range(n) if i not in skip
    ]


# --------------------------------------------------------------------------
# core properties
# --------------------------------------------------------------------------

def test_blocks_are_chronological_and_disjoint():
    p = plan_split(days(60))
    check_plan(p)
    assert p.train[-1] < p.val[0] < p.val[-1] < p.test[0]


def test_train_is_the_past_and_test_is_the_future():
    ds = days(60)
    p = plan_split(ds)
    assert p.train[0] == ds[0], "train must start at the earliest date"
    assert p.test[-1] == ds[-1], "test must end at the latest date"


def test_washout_gap_is_respected_in_calendar_days():
    for gap in (0, 1, 3, 7):
        p = plan_split(days(90), gap_days=gap)
        check_plan(p)
        if gap:
            g1 = (date.fromisoformat(p.val[0]) - date.fromisoformat(p.train[-1])).days
            g2 = (date.fromisoformat(p.test[0]) - date.fromisoformat(p.val[-1])).days
            assert g1 > gap, f"train->val gap {g1} <= {gap}"
            assert g2 > gap, f"val->test gap {g2} <= {gap}"


def test_no_date_appears_in_two_splits():
    p = plan_split(days(60))
    allv = p.train + p.val + p.test
    assert len(allv) == len(set(allv)), "a date is in more than one split"


def test_excluded_dates_are_in_no_split():
    p = plan_split(days(60), gap_days=3)
    assert p.excluded, "a 3-day washout on dense daily data must exclude dates"
    inside = set(p.train + p.val + p.test)
    assert not (set(p.excluded) & inside), "a washout date leaked into a split"


def test_every_date_is_either_split_or_excluded():
    ds = days(60)
    p = plan_split(ds)
    accounted = set(p.train + p.val + p.test + p.excluded)
    assert accounted == set(ds), "dates went missing entirely"


def test_deterministic_on_repeat_calls():
    ds = days(45)
    a, b = plan_split(ds), plan_split(ds)
    assert a.as_dict() == b.as_dict()
    # and insensitive to the order the dates arrive in
    c = plan_split(list(reversed(ds)))
    assert a.as_dict() == c.as_dict()


def test_duplicate_input_dates_are_collapsed():
    ds = days(40)
    assert plan_split(ds).as_dict() == plan_split(ds + ds).as_dict()


def test_gaps_in_capture_do_not_need_extra_washout():
    """A real calendar hole already separates the blocks."""
    ds = days(60, skip=set(range(40, 50)))     # ten missed days mid-series
    p = plan_split(ds, gap_days=3)
    check_plan(p)


# --------------------------------------------------------------------------
# failing loudly
# --------------------------------------------------------------------------

def test_too_few_days_raises_with_a_useful_message():
    for n in (0, 1, 3, 5):
        try:
            plan_split(days(n), gap_days=3)
        except NotEnoughDataError as e:
            msg = str(e)
            assert "keep shooting" in msg.lower(), f"unhelpful message for n={n}: {msg}"
        else:
            raise AssertionError(f"n={n} produced a split instead of failing")


def test_does_not_silently_produce_an_empty_split():
    for n in range(0, 30):
        try:
            p = plan_split(days(n), gap_days=DEFAULT_GAP_DAYS)
        except NotEnoughDataError:
            continue
        assert p.train and p.val and p.test, f"degenerate split at n={n}"
        check_plan(p)


def test_bad_arguments_rejected():
    for bad in ([-1], [0, 0, 0]):
        try:
            plan_split(days(60), ratios=tuple(bad * 3)[:3])
        except (ValueError, NotEnoughDataError):
            pass
        else:
            raise AssertionError(f"accepted bad ratios {bad}")
    try:
        plan_split(days(60), gap_days=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a negative gap")


def test_malformed_date_rejected():
    try:
        plan_split(["2026-01-01", "not-a-date"])
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a malformed date")


# --------------------------------------------------------------------------
# structural: a random split must be impossible, not merely discouraged
# --------------------------------------------------------------------------

def test_module_contains_no_randomness():
    """Structural, over the AST - not a grep.

    A regex over the source would trip on the module docstring, which has to
    be free to explain WHY there is no randomness. What matters is that no
    executable statement imports an RNG or calls one.
    """
    import ast

    src = (Path(__file__).resolve().parent.parent / "src" / "splits.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    banned_mods = {"random", "numpy.random", "secrets"}
    banned_attrs = {"shuffle", "sample", "choice", "permutation", "randint", "rand"}
    banned_names = {"random_state", "seed"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] not in banned_mods, (
                    f"src/splits.py imports {a.name!r}"
                )
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned_mods, (
                f"src/splits.py imports from {node.module!r}"
            )
        elif isinstance(node, ast.Attribute):
            assert node.attr not in banned_attrs, (
                f"src/splits.py calls .{node.attr}() - a random split must "
                "require writing a new function, not passing a flag"
            )
        elif isinstance(node, ast.Name):
            assert node.id not in banned_names, (
                f"src/splits.py references {node.id!r}"
            )
        elif isinstance(node, ast.arg):
            assert node.arg not in banned_names | {"shuffle"}, (
                f"src/splits.py takes a {node.arg!r} argument"
            )


def test_no_shuffle_parameter_in_the_public_api():
    import inspect

    from src import splits

    for fn in (splits.plan_split, splits.build_split):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"shuffle", "random_state", "seed", "random"}), (
            f"{fn.__name__} exposes a randomness knob"
        )


# --------------------------------------------------------------------------
# database-backed
# --------------------------------------------------------------------------

def _fixture_db(n_days: int = 60, per_day: int = 3) -> sqlite3.Connection:
    from src.server import db as _db

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(_db.SCHEMA)
    for i, d in enumerate(days(n_days)):
        cur = con.execute(
            "INSERT INTO sessions (session_date, captured_at, device, kind) "
            "VALUES (?,?,?,'session')",
            (d, d + "T08:00:00", "iphone15"),
        )
        sid = cur.lastrowid
        for p in ("frontal", "left60", "right60")[:per_day]:
            con.execute(
                "INSERT INTO images (session_id, pose, lens, path, sha256) "
                "VALUES (?,?,?,?,?)",
                (sid, p, "main", f"/x/{d}_{p}.jpg", f"sha{i}{p}"),
            )
    # Calibration shots that must never enter a split.
    cur = con.execute(
        "INSERT INTO sessions (session_date, captured_at, device, kind) "
        "VALUES ('2026-01-05','2026-01-05T08:00:00','iphone15','calib')"
    )
    con.execute(
        "INSERT INTO images (session_id, pose, lens, path, sha256) "
        "VALUES (?,?,?,?,?)", (cur.lastrowid, "left60", "main", "/x/cal.jpg", "shacal"),
    )
    con.commit()
    return con


def test_build_split_partitions_images_without_overlap():
    con = _fixture_db()
    s = build_split(con, name="unittest", persist=False)
    allimg = s.images["train"] + s.images["val"] + s.images["test"]
    assert len(allimg) == len(set(allimg)), "an image is in two splits"
    assert all(c["images"] > 0 for c in s.counts.values())


def test_calibration_images_never_enter_a_split():
    con = _fixture_db()
    s = build_split(con, name="unittest", persist=False)
    cal = {
        r[0] for r in con.execute(
            "SELECT i.id FROM images i JOIN sessions s ON s.id=i.session_id "
            "WHERE s.kind='calib'"
        )
    }
    assert cal, "fixture should contain calibration images"
    inside = set(s.images["train"] + s.images["val"] + s.images["test"])
    assert not (cal & inside), "a calibration image leaked into a split"


def test_build_split_is_deterministic():
    con = _fixture_db()
    a = build_split(con, name="unittest", persist=False)
    b = build_split(con, name="unittest", persist=False)
    assert a.dates == b.dates and a.images == b.images


def test_build_split_on_a_young_dataset_fails_loudly():
    con = _fixture_db(n_days=4)
    try:
        build_split(con, name="unittest", persist=False)
    except NotEnoughDataError as e:
        assert "keep shooting" in str(e).lower()
    else:
        raise AssertionError("built a split from 4 days")


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
    print("splits")
    print("-" * 62)
    raise SystemExit(_run())
