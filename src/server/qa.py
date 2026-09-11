"""Per-upload quality checks - reject in the session, not in week six.

Runs on the bytes as received. Nothing here modifies the image; the array is
decoded into memory, measured, and thrown away. The stored file is always the
exact bytes the phone sent.

Checks:
  blur          variance of Laplacian on the luma channel
  exposure      fraction of clipped highlight and shadow pixels
  fiducial      was the 30 mm ArUco marker found, and at what px/mm
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


@dataclass
class QAResult:
    verdict: str = "FAIL"
    reasons: list[str] | None = None
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
        return d


def _decode(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def check(data: bytes, *, require_fiducial: bool = True) -> QAResult:
    r = QAResult()
    reasons: list[str] = []

    img = _decode(data)
    if img is None:
        r.reasons = ["could not decode image - is it actually a JPEG?"]
        return r

    r.height_px, r.width_px = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- blur ---------------------------------------------------------
    # Measured on a downscaled copy so the threshold means the same thing
    # regardless of whether the phone sent 12 MP or 48 MP.
    s = 2000.0 / max(gray.shape)
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    r.lap_var = float(cv2.Laplacian(g, cv2.CV_64F).var())
    if r.lap_var < config.QA_MIN_LAPLACIAN_VAR:
        reasons.append(f"blurry: laplacian var {r.lap_var:.0f} < {config.QA_MIN_LAPLACIAN_VAR:.0f}")

    # --- exposure -----------------------------------------------------
    r.mean_luma = float(gray.mean())
    r.clipped_high_frac = float((gray >= HI).mean())
    r.clipped_low_frac = float((gray <= LO).mean())
    if r.clipped_high_frac > config.QA_MAX_CLIPPED_HIGH:
        reasons.append(
            f"blown highlights: {r.clipped_high_frac:.1%} of pixels clipped "
            f"(limit {config.QA_MAX_CLIPPED_HIGH:.1%}) - move the lamp or diffuse it"
        )
    if r.clipped_low_frac > config.QA_MAX_CLIPPED_LOW:
        reasons.append(
            f"crushed shadows: {r.clipped_low_frac:.1%} of pixels clipped "
            f"(limit {config.QA_MAX_CLIPPED_LOW:.1%})"
        )

    # --- fiducial -----------------------------------------------------
    f = fiducial.detect_fiducial(img)
    r.aruco_found = f.found
    if f.found and f.px_per_mm:
        r.aruco_px_per_mm = round(float(f.px_per_mm), 3)
        r.aruco_edge_spread_pct = round(float(f.edge_spread_pct), 2)
        r.lesion_px_at_1p5mm = round(1.5 * float(f.px_per_mm), 1)
        if r.lesion_px_at_1p5mm < 6:
            reasons.append(
                f"scale too low: a 1.5 mm lesion spans only {r.lesion_px_at_1p5mm:.1f} px - "
                "move closer or use the 2x lens"
            )
    elif require_fiducial:
        reasons.append(f"no ArUco marker found ({f.note}) - without it there is no mm scale")

    r.reasons = reasons
    r.verdict = "PASS" if not reasons else "FAIL"
    return r


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
