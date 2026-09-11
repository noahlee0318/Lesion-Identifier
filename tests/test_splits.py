"""Tests for the date-based split.

The point of these is not that the arithmetic works - it is that the split is
structurally incapable of leaking. Leakage here produces excellent metrics
that mean nothing, and nothing else in the pipeline would catch it.

Runs standalone (no pytest needed):
    python -m tests.test_splits
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.splits import (  # noqa: E402
    DEFAULT_GAP_DAYS,
    DEFAULT_RATIOS,
    DEFAULT_TEST_DAYS,
    DEFAULT_VAL_DAYS,
    NotEnoughDataError,
    Split,
    SplitExistsError,
    SplitIntegrityError,
    build_split,
    check_plan,
    content_hash_of,
    existing_splits,
    load_split,
    min_days_for_window,
    plan_split,
    save_split,
    split_path,
    splits_dir,
    validate_split,
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
# fixed trailing windows - the default boundary rule
# --------------------------------------------------------------------------

def test_window_mode_is_the_default():
    p = plan_split(days(60))
    assert p.mode == "window"
    assert p.ratios is None, (
        "a window split has no requested ratios; recording 0.70/0.15/0.15 "
        "would be a claim about the split that is not true of it"
    )


def test_test_block_is_the_last_fourteen_calendar_days():
    ds = days(60)
    p = plan_split(ds)
    last = date.fromisoformat(ds[-1])
    assert p.test[-1] == ds[-1]
    assert len(p.test) == DEFAULT_TEST_DAYS
    assert (last - date.fromisoformat(p.test[0])).days == DEFAULT_TEST_DAYS - 1


def test_val_block_is_the_fourteen_days_before_the_washout():
    p = plan_split(days(60))
    assert len(p.val) == DEFAULT_VAL_DAYS
    check_plan(p)


def test_test_block_stays_the_same_size_as_the_dataset_grows():
    """The whole point of the change: a stable-size, always-current held-out set."""
    sizes = []
    for n in (40, 60, 120, 365):
        p = plan_split(days(n))
        sizes.append(len(p.test))
        assert p.test[-1] == days(n)[-1], "test must always end at the newest day"
    assert len(set(sizes)) == 1, f"test block size drifted with dataset size: {sizes}"
    assert sizes[0] == DEFAULT_TEST_DAYS


def test_ratio_mode_test_block_grows_without_bound():
    """The behaviour being replaced, kept in a test so the contrast is on record."""
    small = plan_split(days(40), mode="ratio")
    big = plan_split(days(365), mode="ratio")
    assert len(big.test) > 4 * len(small.test), (
        "ratio mode is supposed to grow the test set with the dataset; if it "
        "no longer does, this test is describing the wrong thing"
    )


def test_train_absorbs_everything_earlier():
    ds = days(120)
    p = plan_split(ds)
    assert p.train[0] == ds[0]
    assert len(p.train) > len(p.test) + len(p.val)


def test_window_mode_needs_about_thirty_five_days():
    need = min_days_for_window()
    assert need == 35, need
    for n in range(1, need):
        try:
            plan_split(days(n))
        except NotEnoughDataError as e:
            assert "keep shooting" in str(e).lower()
        else:
            raise AssertionError(f"built a fixed-window split from {n} days")
    p = plan_split(days(need))
    check_plan(p)
    assert len(p.train) == 1, "the 35th day should be the first that works"


def test_the_not_enough_message_names_the_number_of_days():
    try:
        plan_split(days(20))
    except NotEnoughDataError as e:
        msg = str(e)
        assert "35" in msg, msg
        assert "keep shooting" in msg.lower()
    else:
        raise AssertionError("20 days produced a split")


def test_window_mode_handles_sparse_capture_honestly():
    """Three sessions a week: the last 14 CALENDAR days hold ~6 of them."""
    d0 = date(2026, 1, 1)
    ds = [(d0 + timedelta(days=i)).isoformat() for i in range(120) if i % 7 in (0, 2, 4)]
    p = plan_split(ds)
    check_plan(p)
    assert len(p.test) < DEFAULT_TEST_DAYS, "a calendar window cannot invent sessions"
    last = date.fromisoformat(ds[-1])
    assert all((last - date.fromisoformat(s)).days < DEFAULT_TEST_DAYS for s in p.test)


def test_window_mode_respects_the_washout():
    for gap in (0, 1, 3, 7):
        p = plan_split(days(120), gap_days=gap)
        check_plan(p)


def test_window_sizes_are_validated():
    for kw in ({"test_days": 0}, {"val_days": -1}):
        try:
            plan_split(days(60), **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {kw}")


def test_unknown_mode_rejected():
    try:
        plan_split(days(60), mode="random")
    except ValueError as e:
        assert "mode" in str(e)
    else:
        raise AssertionError("accepted an unknown mode")


def test_ratio_mode_is_still_reachable_and_unchanged():
    p = plan_split(days(60), mode="ratio", ratios=(0.70, 0.15, 0.15))
    check_plan(p)
    assert p.mode == "ratio" and p.ratios == (0.70, 0.15, 0.15)
    assert len(p.train) == 42, "the proportional rule changed behaviour"


# --------------------------------------------------------------------------
# achieved ratios - what the split IS, not what was asked for
# --------------------------------------------------------------------------

def test_achieved_ratios_differ_from_the_requested_ones():
    p = plan_split(days(60), mode="ratio", ratios=(0.70, 0.15, 0.15))
    assert p.ratios == (0.70, 0.15, 0.15)
    a = p.achieved_ratios
    assert abs(a[0] - 0.778) < 0.01, a
    assert abs(a[1] - 0.111) < 0.01, a
    assert a != p.ratios, (
        "the washout eats the front of val and test, so the achieved split is "
        "not the requested one - recording only the request is misleading"
    )


def test_achieved_ratios_sum_to_one():
    """To within the 4-decimal rounding they are stored at, for readability."""
    for n in (40, 60, 200):
        for mode in ("window", "ratio"):
            a = plan_split(days(n), mode=mode).achieved_ratios
            assert abs(sum(a) - 1.0) < 1e-3, (n, mode, a)


def test_achieved_ratios_match_the_day_counts():
    p = plan_split(days(90))
    n = len(p.train) + len(p.val) + len(p.test)
    assert abs(p.achieved_ratios[0] - len(p.train) / n) < 1e-4
    assert abs(p.achieved_ratios[2] - len(p.test) / n) < 1e-4


def test_split_records_both_requested_and_achieved():
    con = _fixture_db(n_days=60)
    s = build_split(con, name="unittest", persist=False, mode="ratio")
    assert s.ratios == DEFAULT_RATIOS
    assert abs(sum(s.achieved_ratios) - 1.0) < 1e-6
    assert s.achieved_ratios != s.ratios


# --------------------------------------------------------------------------
# split files are write-once
# --------------------------------------------------------------------------

def _temp_splits_dir():
    """Point splits_dir() at a scratch directory for the duration of a test."""
    import tempfile

    from src import config as _config

    d = Path(tempfile.mkdtemp(prefix="atlas_splits_"))
    old = _config.DERIVED_DIR
    _config.DERIVED_DIR = d
    return d, old


def _restore(old):
    from src import config as _config

    _config.DERIVED_DIR = old


def test_save_refuses_to_overwrite_an_existing_split():
    _, old = _temp_splits_dir()
    try:
        con = _fixture_db(n_days=60)
        first = build_split(con, name="v1")
        con2 = _fixture_db(n_days=80)                 # more data, same name
        try:
            build_split(con2, name="v1")
        except SplitExistsError as e:
            msg = str(e)
            assert first.created_at in msg, "the refusal does not say when the old one was made"
            assert "img" in msg, "the refusal does not say what is in the old one"
            assert "--force" in msg and "new name" in msg.lower().replace("pick a new name", "new name")
        else:
            raise AssertionError(
                "rebuilding v1 silently destroyed the split last month's metrics "
                "were computed against"
            )
    finally:
        _restore(old)


def test_force_destroys_but_says_so():
    _, old = _temp_splits_dir()
    try:
        con = _fixture_db(n_days=60)
        build_split(con, name="v1")
        before = existing_splits("v1")
        assert len(before) == 1

        con2 = _fixture_db(n_days=80)
        s2 = build_split(con2, name="v1", force=True)
        after = existing_splits("v1")
        assert len(after) == 1, "force left two files claiming one name"
        assert after[0] == split_path(s2)
        assert after[0] != before[0], "force did not actually replace anything"
    finally:
        _restore(old)


def test_rebuilding_unchanged_data_is_not_a_conflict():
    """Same content, same name: the same split, not a collision."""
    _, old = _temp_splits_dir()
    try:
        con = _fixture_db(n_days=60)
        a = build_split(con, name="v1")
        pa = split_path(a)
        stamp = json.loads(pa.read_text(encoding="utf-8"))["created_at"]

        con2 = _fixture_db(n_days=60)
        b = build_split(con2, name="v1")              # must not raise
        assert split_path(b) == pa
        assert len(existing_splits("v1")) == 1
        kept = json.loads(pa.read_text(encoding="utf-8"))["created_at"]
        assert kept == stamp, "the original creation date was overwritten"
    finally:
        _restore(old)


def test_filename_carries_a_content_hash():
    _, old = _temp_splits_dir()
    try:
        con = _fixture_db(n_days=60)
        s = build_split(con, name="v1", persist=False)
        p = split_path(s)
        assert p.name.startswith("v1.") and p.name.endswith(".json")
        assert p.stem.split(".")[1] == content_hash_of(s)
        assert len(p.stem.split(".")[1]) == 8
    finally:
        _restore(old)


def test_different_data_gives_a_different_filename():
    _, old = _temp_splits_dir()
    try:
        a = build_split(_fixture_db(n_days=60), name="v1", persist=False)
        b = build_split(_fixture_db(n_days=80), name="v1", persist=False)
        assert split_path(a) != split_path(b), (
            "two different splits would share a filename - the name would "
            "silently mean two different things"
        )
    finally:
        _restore(old)


def test_hash_ignores_the_timestamp():
    con = _fixture_db(n_days=60)
    a = build_split(con, name="v1", persist=False)
    b = build_split(con, name="v1", persist=False)
    assert a.created_at is not None
    assert content_hash_of(a) == content_hash_of(b)


# --------------------------------------------------------------------------
# load validates
# --------------------------------------------------------------------------

def test_load_round_trips():
    _, old = _temp_splits_dir()
    try:
        con = _fixture_db(n_days=60)
        s = build_split(con, name="v1")
        back = load_split("v1")
        assert back.dates == s.dates
        assert back.images == s.images
        assert tuple(back.achieved_ratios) == tuple(s.achieved_ratios)
        assert back.mode == s.mode
        # also loadable by full stem
        assert load_split(split_path(s).stem).dates == s.dates
    finally:
        _restore(old)


def test_a_hand_edited_split_fails_loudly_on_load():
    """The failure this module exists to prevent, arriving through the back door."""
    _, old = _temp_splits_dir()
    try:
        con = _fixture_db(n_days=60)
        s = build_split(con, name="v1")
        p = split_path(s)
        d = json.loads(p.read_text(encoding="utf-8"))
        # Move one test day into train: leakage, and the washout is now violated.
        stolen = d["dates"]["test"][0]
        d["dates"]["train"].append(stolen)
        d["counts"]["train"]["days"] += 1
        p.write_text(json.dumps(d), encoding="utf-8")

        try:
            load_split("v1")
        except SplitIntegrityError as e:
            assert "invariant" in str(e) or "achieved_ratios" in str(e), str(e)
        else:
            raise AssertionError(
                "a leaking split file loaded cleanly and would have produced "
                "fake metrics"
            )
    finally:
        _restore(old)


def test_a_truncated_split_fails_loudly_on_load():
    _, old = _temp_splits_dir()
    try:
        con = _fixture_db(n_days=60)
        s = build_split(con, name="v1")
        p = split_path(s)
        d = json.loads(p.read_text(encoding="utf-8"))
        d["dates"]["val"] = []                        # truncated file
        p.write_text(json.dumps(d), encoding="utf-8")
        try:
            load_split("v1")
        except SplitIntegrityError:
            pass
        else:
            raise AssertionError("an empty val block loaded without complaint")
    finally:
        _restore(old)


def test_inconsistent_counts_are_caught():
    _, old = _temp_splits_dir()
    try:
        con = _fixture_db(n_days=60)
        s = build_split(con, name="v1")
        p = split_path(s)
        d = json.loads(p.read_text(encoding="utf-8"))
        d["counts"]["test"]["images"] = 999999
        p.write_text(json.dumps(d), encoding="utf-8")
        try:
            load_split("v1")
        except SplitIntegrityError as e:
            assert "counts" in str(e)
        else:
            raise AssertionError("a wrong image count loaded without complaint")
    finally:
        _restore(old)


def test_an_image_in_two_blocks_is_caught():
    _, old = _temp_splits_dir()
    try:
        con = _fixture_db(n_days=60)
        s = build_split(con, name="v1")
        p = split_path(s)
        d = json.loads(p.read_text(encoding="utf-8"))
        leaked = d["images"]["test"][0]
        d["images"]["train"].append(leaked)
        d["counts"]["train"]["images"] += 1
        p.write_text(json.dumps(d), encoding="utf-8")
        try:
            load_split("v1")
        except SplitIntegrityError as e:
            assert "both" in str(e)
        else:
            raise AssertionError("an image in train AND test loaded without complaint")
    finally:
        _restore(old)


def test_ambiguous_name_is_an_error_not_a_guess():
    _, old = _temp_splits_dir()
    try:
        a = build_split(_fixture_db(n_days=60), name="v1")
        b = build_split(_fixture_db(n_days=80), name="v1", force=True)
        # Put the first one back alongside the second, simulating a copied file.
        split_path(a).write_text(json.dumps(a.to_dict()), encoding="utf-8")
        assert len(existing_splits("v1")) == 2
        try:
            load_split("v1")
        except ValueError as e:
            assert "claim the split name" in str(e)
        else:
            raise AssertionError("an ambiguous name silently resolved to one file")
        # The full stem is unambiguous and still works.
        assert load_split(split_path(b).stem).dates == b.dates
    finally:
        _restore(old)


def test_validate_accepts_a_healthy_split():
    con = _fixture_db(n_days=60)
    validate_split(build_split(con, name="unittest", persist=False))


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
