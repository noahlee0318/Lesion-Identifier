"""Central paths and constants for the Lesion Atlas.

DATA_ROOT lives OUTSIDE the repo on purpose, as a sibling of it:

    C:\\LesionAtlas\\
        data\\           <- DATA_ROOT, never committed
        lesion-atlas\\   <- this repo

The whole project sits on a plain local path, deliberately off any
cloud-synced folder. It previously lived under OneDrive\\Desktop, which would
have auto-uploaded every photo; OneDrive has no reliable per-subfolder upload
exclusion, so the fix was to move the project out rather than carve an
exception.

Keeping data outside the repo still matters now that nothing is synced: it
means no git operation, no clone, and no accidental `git add -f` can reach an
image. Override with the LESION_ATLAS_DATA environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- repo-relative ---------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO / "models" / "weights"
REPORTS_DIR = REPO / "reports"
DOCS_DIR = REPO / "docs"
LOGS_DIR = REPO / "logs"
FIXTURES_DIR = REPO / "tests" / "fixtures"

FACE_LANDMARKER_TASK = WEIGHTS_DIR / "face_landmarker.task"

# --- data root (outside the repo, never on a synced path) -------------------
DATA_ROOT = Path(os.environ.get("LESION_ATLAS_DATA", r"C:\LesionAtlas\data"))
RAW_DIR = DATA_ROOT / "raw"
CALIB_DIR = RAW_DIR / "calib"
DERIVED_DIR = DATA_ROOT / "derived"
CONTROLPOINTS_DIR = DERIVED_DIR / "controlpoints"
ATLAS_DIR = DATA_ROOT / "atlas"
DB_PATH = DATA_ROOT / "atlas.sqlite3"

# --- locked domain constants (CLAUDE.md - do not re-litigate) ---------------
POSES = ("frontal", "left60", "right60")
LENSES = ("main", "2x")

REGIONS = (
    "forehead",
    "chin_perioral",
    "cheek_l",
    "cheek_r",
    "temple_sideburn_l",
    "temple_sideburn_r",
)

# Which pose OWNS (counts) which region. Everything else is validation-only.
REGION_OWNER = {
    "forehead": "frontal",
    "chin_perioral": "frontal",
    "cheek_l": "left60",
    "temple_sideburn_l": "left60",
    "cheek_r": "right60",
    "temple_sideburn_r": "right60",
}

# --- fiducial ---------------------------------------------------------------
ARUCO_DICT_NAME = "DICT_4X4_50"
ARUCO_ID = 0
ARUCO_SIDE_MM = 30.0

# --- phase 0 gate (build-plan p4) ------------------------------------------
GATE_MEDIAN_MM = 1.5
GATE_P95_MM = 3.0

# --- QA thresholds (provisional; retune once real photos exist) ------------
QA_MIN_LAPLACIAN_VAR = 100.0
QA_MAX_CLIPPED_HIGH = 0.02
QA_MAX_CLIPPED_LOW = 0.05


def ensure_dirs() -> None:
    """Create the data tree. Safe to call repeatedly."""
    for d in (
        RAW_DIR,
        CALIB_DIR,
        DERIVED_DIR,
        CONTROLPOINTS_DIR,
        ATLAS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def torch_home() -> str:
    """Point torch.hub at the in-repo weight cache."""
    os.environ.setdefault("TORCH_HOME", str(WEIGHTS_DIR))
    return os.environ["TORCH_HOME"]
