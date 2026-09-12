"""Per-upload quality checks - reject in the session, not in week six.

Runs on the bytes as received. Nothing here modifies the image; the array is
decoded into memory, measured, and thrown away. The stored file is always the
exact bytes the phone sent.

Checks:
  blur          variance of Laplacian on the luma channel
  exposure      fraction of clipped highlight and shadow pixels
  fiducial      was the 30 mm ArUco marker found, and at what px/mm

=============================================================================
FAILURE MESSAGES ARE WRITTEN FOR A PERSON HOLDING A PHONE, NOT FOR A LOG

A rejection is only useful if it is acted on in the same sitting - while the
lamp and the tape are still where they were. Someone reading "laplacian var
47 < 100" at 7am has to go and look up what that means, and will instead
shrug and upload it anyway. That is how a bad frame gets into a longitudinal
series, which is the exact thing this module exists to stop.

So every problem is a `Problem`, with three separate fields:

    what    what is wrong with the photo, in words, no units of measurement
            and no algorithm names. "Out of focus."
    fix     what to physically DO about it, right now. This is the field
            that determines whether the reshoot happens.
    detail  the measurement, for diagnosis. Shown small, behind a toggle,
            and never the first thing read.

`detail` is not optional and is never dropped. When a threshold turns out to
be tuned wrong - and these are provisional until real photos exist - the
number is the only way to find out. Plain English is about ORDERING, not
about hiding the evidence.
=============================================================================
"""

from __future__ import annotations

import io
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .. import config, fiducial
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src import config, fiducial

HI, LO = 250, 5          # 8-bit clipping thresholds

# A 1.5 mm lesion needs at least this many pixels across it to be measurable.
MIN_LESION_PX = 6

# How far past the marker's own corners to mask, as a multiple of its edge
# length. The printed quiet zone is 8 mm around a 30 mm marker, so ~0.27; 0.35
# covers that plus the paper's margin and any perspective.
MARKER_MASK_MARGIN = 0.35


@dataclass
class Problem:
    """One reason a photo was rejected, in three registers.

    See the module docstring. `what` and `fix` are read by a person; `detail`
    is read when a threshold is being questioned.
    """

    what: str
    fix: str
    detail: str = ""

    def as_sentence(self) -> str:
        """Plain-English one-liner, for logs and for anything reading `reasons`."""
        return f"{self.what} {self.fix}".strip()

    def to_dict(self) -> dict[str, str]:
        return {"what": self.what, "fix": self.fix, "detail": self.detail}


@dataclass
class QAResult:
    verdict: str = "FAIL"
    reasons: list[str] | None = None
    problems: list[Problem] | None = None
    lap_var: float = float("nan")
    clipped_high_frac: float = float("nan")
    clipped_low_frac: float = float("nan")
    mean_luma: float = float("nan")
    aruco_found: bool = False
    aruco_px_per_mm: float | None = None
    aruco_edge_spread_pct: float | None = None
    width_px: int = 0
    height_px: int = 0
    lesion_px_at_1p5mm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reasons"] = self.reasons or []
        d["problems"] = [pr.to_dict() for pr in (self.problems or [])]
        return d


def _decode(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _mask_out_marker(gray: np.ndarray, f) -> np.ndarray:
    """The luma values OUTSIDE the fiducial, as a flat array.

    Returns every pixel when the marker was not found - there is nothing to
    exclude, and a photo with no marker is already being rejected for that.
    """
    if not f.found or len(f.corners) != 4:
        return gray.ravel()

    quad = np.asarray(f.corners, np.float64)
    centre = quad.mean(axis=0)
    grown = centre + (quad - centre) * (1.0 + 2.0 * MARKER_MASK_MARGIN)

    keep = np.ones(gray.shape, np.uint8)
    cv2.fillConvexPoly(keep, grown.round().astype(np.int32), 0)
    out = gray[keep.astype(bool)]
    # If the "marker" somehow covers the whole frame, fall back rather than
    # reporting exposure on nothing.
    return out if out.size >= gray.size // 10 else gray.ravel()


def _finish(r: QAResult, problems: list[Problem]) -> QAResult:
    r.problems = problems
    r.reasons = [pr.as_sentence() for pr in problems]
    r.verdict = "PASS" if not problems else "FAIL"
    return r


def check(data: bytes, *, require_fiducial: bool = True) -> QAResult:
    r = QAResult()
    problems: list[Problem] = []

    img = _decode(data)
    if img is None:
        return _finish(r, [Problem(
            what="This file is not a photo the laptop can open.",
            fix="On the phone: Settings -> Camera -> Formats -> Most Compatible, "
                "then take the shot again. That makes it save JPEG instead of HEIC.",
            detail="the image decoder returned nothing for these bytes",
        )])

    r.height_px, r.width_px = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the marker FIRST: the exposure check below has to exclude it.
    f = fiducial.detect_fiducial(img)
    r.aruco_found = f.found

    # --- blur ---------------------------------------------------------
    # Measured on a downscaled copy so the threshold means the same thing
    # regardless of whether the phone sent 12 MP or 48 MP.
    s = 2000.0 / max(gray.shape)
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    r.lap_var = float(cv2.Laplacian(g, cv2.CV_64F).var())
    if r.lap_var < config.QA_MIN_LAPLACIAN_VAR:
        problems.append(Problem(
            what="Out of focus.",
            fix="Tap the screen on your cheek to focus there, hold still, shoot again.",
            detail=f"sharpness {r.lap_var:.0f}, needs {config.QA_MIN_LAPLACIAN_VAR:.0f} or more",
        ))

    # --- exposure -----------------------------------------------------
    # Measured with the MARKER MASKED OUT. The marker is white paper and black
    # ink by design and is deliberately in every frame: at 26 px/mm a 30 mm
    # marker plus its 8 mm quiet zone is ~12% of a 4032x3024 photo, and the
    # white alone is ~6.5% - over triple the 2% "too bright" limit. Measuring
    # it would fail essentially every correctly-exposed photo and blame the
    # lamp, which is worse than useless: it trains the reader to override the
    # QA verdict, and then the verdict stops catching the real thing.
    #
    # This check is about whether SKIN is readable. So it measures skin.
    skin = _mask_out_marker(gray, f)
    r.mean_luma = float(gray.mean())
    r.clipped_high_frac = float((skin >= HI).mean()) if skin.size else float("nan")
    r.clipped_low_frac = float((skin <= LO).mean()) if skin.size else float("nan")
    if r.clipped_high_frac > config.QA_MAX_CLIPPED_HIGH:
        problems.append(Problem(
            what="Too bright in places - parts of your skin are pure white with no "
                 "detail left, so anything there cannot be seen at all.",
            fix="Move the lamp further back, or tape a sheet of paper over it to "
                "soften it. Do not point it straight at your face.",
            detail=f"{r.clipped_high_frac:.1%} of the photo is pure white, "
                   f"limit {config.QA_MAX_CLIPPED_HIGH:.1%}",
        ))
    if r.clipped_low_frac > config.QA_MAX_CLIPPED_LOW:
        problems.append(Problem(
            what="Too dark in places - parts of the photo are solid black with no "
                 "detail left.",
            fix="Bring the lamp round towards the shadow side of your face, or add "
                "a second light source.",
            detail=f"{r.clipped_low_frac:.1%} of the photo is solid black, "
                   f"limit {config.QA_MAX_CLIPPED_LOW:.1%}",
        ))

    # --- fiducial -----------------------------------------------------
    if f.found and f.px_per_mm:
        r.aruco_px_per_mm = round(float(f.px_per_mm), 3)
        r.aruco_edge_spread_pct = round(float(f.edge_spread_pct), 2)
        r.lesion_px_at_1p5mm = round(1.5 * float(f.px_per_mm), 1)
        if r.lesion_px_at_1p5mm < MIN_LESION_PX:
            problems.append(Problem(
                what="Too far away. The smallest spots this is meant to track come "
                     "out too small to measure at this distance.",
                fix="Step closer, or switch to the 2x lens and back up to about 55 cm.",
                detail=f"a 1.5 mm spot is {r.lesion_px_at_1p5mm:.1f} pixels across, "
                       f"needs {MIN_LESION_PX}",
            ))
    elif require_fiducial:
        problems.append(Problem(
            what="The printed marker is not in the photo, or is not readable.",
            fix="Get the whole black square in the shot, in focus, and not glaring "
                "under the lamp. Everything is measured in millimetres and that "
                "square is the only thing that says how big a millimetre is.",
            detail=f"marker detection: {f.note}",
        ))

    return _finish(r, problems)


def sha256_of(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def read_exif(data: bytes) -> dict:
    """Read EXIF from the received bytes. Read-only; nothing is written back."""
    out: dict[str, Any] = {}
    try:
        from PIL import Image, ExifTags

        with Image.open(io.BytesIO(data)) as im:
            out["pil_size"] = list(im.size)
            out["pil_format"] = im.format
            exif = im.getexif()
            if exif:
                for k, v in exif.items():
                    name = ExifTags.TAGS.get(k, str(k))
                    if isinstance(v, bytes):
                        v = f"<{len(v)} bytes>"
                    out[name] = str(v)[:300]
                try:
                    for k, v in exif.get_ifd(0x8769).items():
                        name = ExifTags.TAGS.get(k, str(k))
                        if isinstance(v, bytes):
                            v = f"<{len(v)} bytes>"
                        out[name] = str(v)[:300]
                except Exception:  # noqa: BLE001
                    pass
    except Exception as exc:  # noqa: BLE001
        out["_exif_error"] = repr(exc)
    return out
