"""SQLite for the phase 0 ingest server.

Three tables now: sessions, images, regimen_events. The detections / lesions /
observations tiers arrive in phases 4-5; nothing here should have to change
when they do.

The database is a convenience index, not the record of truth. The record of
truth is the image bytes plus its sidecar JSON on disk - so a lost or corrupt
database can be rebuilt by walking data/raw, and nothing irreplaceable lives
only in here.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    from .. import config
except ImportError:  # pragma: no cover - direct execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src import config


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_date    TEXT NOT NULL,               -- YYYY-MM-DD, subject-declared
    captured_at     TEXT NOT NULL,               -- ISO8601 local, server clock
    device          TEXT,
    kind            TEXT NOT NULL DEFAULT 'session',   -- 'session' | 'calib'
    covariates_json TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions(session_date);

CREATE TABLE IF NOT EXISTS images (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    pose              TEXT NOT NULL,             -- frontal | left60 | right60
    lens              TEXT NOT NULL,             -- main | 2x
    device            TEXT,
    path              TEXT NOT NULL UNIQUE,
    sidecar_path      TEXT,
    original_filename TEXT,
    bytes             INTEGER,
    sha256            TEXT NOT NULL,
    width_px          INTEGER,
    height_px         INTEGER,
    exif_json         TEXT,
    qa_json           TEXT,
    qa_verdict        TEXT,                      -- PASS | FAIL
    reg_status        TEXT DEFAULT 'unregistered',
    reg_median_err_mm REAL,
    transform_json    TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_images_session ON images(session_id);
CREATE INDEX IF NOT EXISTS idx_images_sha     ON images(sha256);
CREATE INDEX IF NOT EXISTS idx_images_pose    ON images(pose, lens);

CREATE TABLE IF NOT EXISTS regimen_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_date  TEXT NOT NULL,                   -- YYYY-MM-DD
    change      TEXT NOT NULL,                   -- added | stopped | dose_changed
    product     TEXT NOT NULL,
    detail      TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_regimen_date ON regimen_events(event_date);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = Path(path or config.DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


@contextmanager
def tx(con: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise


# --------------------------------------------------------------------------

def get_or_create_session(
    con: sqlite3.Connection,
    session_date: str,
    captured_at: str,
    device: str | None,
    covariates: dict | None,
    kind: str = "session",
) -> int:
    """One session per (date, kind). Re-uploading a pose joins the same session."""
    row = con.execute(
        "SELECT id, covariates_json FROM sessions WHERE session_date=? AND kind=?",
        (session_date, kind),
    ).fetchone()
    if row:
        # Later covariate answers overwrite earlier ones for the same day.
        if covariates:
            merged = json.loads(row["covariates_json"] or "{}")
            merged.update(covariates)
            con.execute(
                "UPDATE sessions SET covariates_json=? WHERE id=?",
                (json.dumps(merged), row["id"]),
            )
        return int(row["id"])
    cur = con.execute(
        "INSERT INTO sessions (session_date, captured_at, device, kind, covariates_json) "
        "VALUES (?,?,?,?,?)",
        (session_date, captured_at, device, kind, json.dumps(covariates or {})),
    )
    return int(cur.lastrowid)


def insert_image(con: sqlite3.Connection, session_id: int, **kw: Any) -> int:
    cols = (
        "pose", "lens", "device", "path", "sidecar_path", "original_filename",
        "bytes", "sha256", "width_px", "height_px", "exif_json", "qa_json", "qa_verdict",
    )
    vals = [kw.get(c) for c in cols]
    cur = con.execute(
        f"INSERT INTO images (session_id, {', '.join(cols)}) "
        f"VALUES (?{', ?' * len(cols)})",
        [session_id, *vals],
    )
    return int(cur.lastrowid)


def find_by_sha(con: sqlite3.Connection, sha: str) -> sqlite3.Row | None:
    return con.execute("SELECT * FROM images WHERE sha256=?", (sha,)).fetchone()


def add_regimen_event(
    con: sqlite3.Connection, event_date: str, change: str, product: str, detail: str | None
) -> int:
    cur = con.execute(
        "INSERT INTO regimen_events (event_date, change, product, detail) VALUES (?,?,?,?)",
        (event_date, change, product, detail),
    )
    return int(cur.lastrowid)


def recent_regimen(con: sqlite3.Connection, limit: int = 20) -> list[dict]:
    rows = con.execute(
        "SELECT * FROM regimen_events ORDER BY event_date DESC, id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# calibration progress
# --------------------------------------------------------------------------

# Phase 0 calibration, per the build plan and runbook: ONE pose (left 60),
# shot five separate times across at least three days, setting up from scratch
# each time, on BOTH lenses. Five repeats total - not five per day.
CALIB_POSE = "left60"
CALIB_REPEATS = 5
CALIB_MIN_DAYS = 3
CALIB_LENSES = ("main", "2x")


def calibration_progress(con: sqlite3.Connection) -> dict:
    """How much of the 5-repeat calibration set exists, and what is missing.

    A 'repeat' is one setup-from-scratch sitting. Two sittings on the same day
    count as two repeats, but the three-day spread is what actually tests
    reproducibility, so both are tracked and reported separately.
    """
    rows = con.execute(
        """
        SELECT s.session_date AS d, i.lens AS lens, i.qa_verdict AS qa, i.created_at AS t
        FROM images i JOIN sessions s ON s.id = i.session_id
        WHERE s.kind='calib' AND i.pose=?
        ORDER BY i.created_at
        """,
        (CALIB_POSE,),
    ).fetchall()

    by_day: dict[str, dict[str, int]] = {}
    for r in rows:
        d = by_day.setdefault(r["d"], {"main": 0, "2x": 0, "pass": 0, "fail": 0})
        if r["lens"] in d:
            d[r["lens"]] += 1
        d["pass" if r["qa"] == "PASS" else "fail"] += 1

    # A repeat is complete when that sitting has BOTH lenses. We cannot see
    # sittings directly, so the conservative count is min(main, 2x) per day.
    complete = sum(min(v["main"], v["2x"]) for v in by_day.values())
    days = sorted(by_day)

    missing = []
    for d in days:
        v = by_day[d]
        if v["main"] != v["2x"]:
            short = "2x" if v["2x"] < v["main"] else "main"
            missing.append(f"{d}: {abs(v['main'] - v['2x'])} shot(s) missing on {short}")

    remaining = max(0, CALIB_REPEATS - complete)
    days_remaining = max(0, CALIB_MIN_DAYS - len(days))

    if remaining == 0 and days_remaining == 0:
        status, msg = "complete", (
            f"All {CALIB_REPEATS} repeats captured across {len(days)} days. "
            "Run facemesh_check and lock the angle."
        )
    elif remaining == 0:
        status, msg = "spread", (
            f"{CALIB_REPEATS} repeats captured but only over {len(days)} day(s). "
            f"Spread over at least {CALIB_MIN_DAYS} - same-day repeats do not test "
            "whether the pose is reproducible tomorrow."
        )
    else:
        status, msg = "in_progress", (
            f"{complete} of {CALIB_REPEATS} repeats done. {remaining} to go, "
            f"across at least {max(1, days_remaining)} more day(s)."
        )

    return {
        "pose": CALIB_POSE,
        "target_repeats": CALIB_REPEATS,
        "min_days": CALIB_MIN_DAYS,
        "lenses": list(CALIB_LENSES),
        "complete_repeats": complete,
        "remaining_repeats": remaining,
        "days_covered": days,
        "days_remaining": days_remaining,
        "total_images": len(rows),
        "expected_images": CALIB_REPEATS * len(CALIB_LENSES),
        "by_day": by_day,
        "unpaired": missing,
        "status": status,
        "message": msg,
    }


def recent_sessions(con: sqlite3.Connection, limit: int = 10) -> list[dict]:
    rows = con.execute(
        """
        SELECT s.id, s.session_date, s.kind, COUNT(i.id) AS n,
               SUM(CASE WHEN i.qa_verdict='PASS' THEN 1 ELSE 0 END) AS n_pass
        FROM sessions s LEFT JOIN images i ON i.session_id = s.id
        GROUP BY s.id ORDER BY s.session_date DESC, s.id DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]
