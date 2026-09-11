"""Print-ready calibration target, and the detector everything else reads mm from.

Two jobs:

  generate  build a letter-sized sheet (PNG + PDF) carrying one 30 mm ArUco
            marker, a six-patch colour strip, a 100 mm ruler line and crop
            marks, plus a sidecar JSON recording every patch value and
            position in millimetres.

  verify    given a photograph of the printed sheet, detect the marker and
            report measured px/mm and whether the four detected edges agree.

Everything is laid out on an exact 12 px/mm grid (304.8 DPI), so the 30 mm
marker is exactly 360 px and lands on a whole number of ArUco cells (6 cells
x 60 px). No rounding error is baked into the printed artefact.

CLI
    python -m src.fiducial generate
    python -m src.fiducial verify path/to/photo_of_sheet.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from . import config
except ImportError:  # running as a script rather than a module
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config


PX_PER_MM = 12.0          # exact: 304.8 DPI
DPI = PX_PER_MM * 25.4    # 304.8
PAGE_W_MM, PAGE_H_MM = 215.9, 279.4   # US Letter

MARKER_MM = config.ARUCO_SIDE_MM       # 30.0
MARKER_ID = config.ARUCO_ID            # 0
QUIET_MM = 10.0                        # white border the detector needs
PATCH_MM = 20.0
PATCH_GAP_MM = 4.0
RULER_MM = 100.0

# sRGB values. 18% gray is the linear-0.18 mid-gray encoded to sRGB:
#   1.055 * 0.18**(1/2.4) - 0.055  ->  0.4613  ->  118
PATCHES: list[tuple[str, tuple[int, int, int]]] = [
    ("white", (255, 255, 255)),
    ("gray18", (118, 118, 118)),
    ("black", (0, 0, 0)),
    ("red", (255, 0, 0)),
    ("green", (0, 255, 0)),
    ("blue", (0, 0, 255)),
]


def mm(v: float) -> int:
    """Millimetres -> pixels on the print grid."""
    return int(round(v * PX_PER_MM))


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def _font(size_px: int) -> ImageFont.FreeTypeFont:
    """Arial if Windows has it, else matplotlib's bundled DejaVu, else default."""
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ]
    try:
        import matplotlib

        candidates.append(
            str(Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf")
        )
    except Exception:  # noqa: BLE001
        pass
    for c in candidates:
        try:
            return ImageFont.truetype(c, size_px)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def generate(out_dir: Path | None = None) -> dict:
    """Render the calibration sheet to PNG + PDF + sidecar JSON."""
    out_dir = out_dir or (config.REPO / "docs" / "fiducial")
    out_dir.mkdir(parents=True, exist_ok=True)

    W, H = mm(PAGE_W_MM), mm(PAGE_H_MM)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)

    f_big = _font(mm(6))
    f_mid = _font(mm(4.2))
    f_sml = _font(mm(3.2))

    margin = 18.0
    y = margin

    # ---- title -----------------------------------------------------------
    d.text((mm(margin), mm(y)), "LESION ATLAS - CALIBRATION TARGET", font=f_big, fill=(0, 0, 0))
    y += 9
    d.text(
        (mm(margin), mm(y)),
        "PRINT AT 100% SCALE.  NO 'FIT TO PAGE'.  NO 'SHRINK TO MARGINS'.  MATTE PAPER ONLY.",
        font=f_mid,
        fill=(180, 0, 0),
    )
    y += 6
    d.text(
        (mm(margin), mm(y)),
        "Gloss paper will specular under the lamp and destroy the white-balance reference.",
        font=f_sml,
        fill=(90, 90, 90),
    )
    y += 12

    # ---- ArUco marker ----------------------------------------------------
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT_NAME))
    side_px = mm(MARKER_MM)                       # 360 px = 6 cells x 60
    marker = cv2.aruco.generateImageMarker(adict, MARKER_ID, side_px)
    marker_rgb = Image.fromarray(cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB))

    mk_x, mk_y = margin + QUIET_MM, y + QUIET_MM
    img.paste(marker_rgb, (mm(mk_x), mm(mk_y)))

    # Thin registration box around the quiet zone so a miscut is obvious.
    d.rectangle(
        [mm(margin), mm(y), mm(margin + MARKER_MM + 2 * QUIET_MM), mm(y + MARKER_MM + 2 * QUIET_MM)],
        outline=(200, 200, 200),
        width=max(1, mm(0.2)),
    )

    tx = margin + MARKER_MM + 2 * QUIET_MM + 8
    ty = y + 4
    for line, fnt, col in [
        (f"ArUco {config.ARUCO_DICT_NAME}   id = {MARKER_ID}", f_mid, (0, 0, 0)),
        (f"PRINTED SIZE: {MARKER_MM:.1f} mm", f_big, (0, 0, 0)),
        ("measured across the OUTER black edge", f_sml, (90, 90, 90)),
        ("", f_sml, (0, 0, 0)),
        ("Verify with a ruler before you shoot.", f_sml, (180, 0, 0)),
        ("If this is not 30.0 mm, the print scaled", f_sml, (180, 0, 0)),
        ("and every millimetre downstream is wrong.", f_sml, (180, 0, 0)),
    ]:
        d.text((mm(tx), mm(ty)), line, font=fnt, fill=col)
        ty += 6.2

    y += MARKER_MM + 2 * QUIET_MM + 10

    # ---- 100 mm ruler line ----------------------------------------------
    d.text((mm(margin), mm(y)), "100.0 mm reference - independent check on print scale", font=f_sml, fill=(0, 0, 0))
    y += 5.5
    rx0, rx1 = margin, margin + RULER_MM
    d.line([mm(rx0), mm(y + 4), mm(rx1), mm(y + 4)], fill=(0, 0, 0), width=max(1, mm(0.4)))
    for i in range(11):                      # ticks every 10 mm
        x = rx0 + i * 10.0
        h = 3.0 if i % 5 else 5.0
        d.line([mm(x), mm(y + 4 - h), mm(x), mm(y + 4)], fill=(0, 0, 0), width=max(1, mm(0.3)))
    d.text((mm(rx1 + 3), mm(y + 1.5)), "100 mm", font=f_sml, fill=(0, 0, 0))
    y += 12

    # ---- colour patches --------------------------------------------------
    d.text((mm(margin), mm(y)), "COLOUR REFERENCE  (sRGB values recorded in the sidecar JSON)", font=f_sml, fill=(0, 0, 0))
    y += 5.5
    patch_geom = []
    px_ = margin
    for name, rgb in PATCHES:
        d.rectangle([mm(px_), mm(y), mm(px_ + PATCH_MM), mm(y + PATCH_MM)], fill=rgb)
        # Hairline keyline so the white patch has a findable edge on white paper.
        d.rectangle(
            [mm(px_), mm(y), mm(px_ + PATCH_MM), mm(y + PATCH_MM)],
            outline=(128, 128, 128),
            width=max(1, mm(0.15)),
        )
        d.text((mm(px_), mm(y + PATCH_MM + 1.2)), name, font=f_sml, fill=(0, 0, 0))
        patch_geom.append(
            {
                "name": name,
                "srgb": list(rgb),
                "x_mm": round(px_, 3),
                "y_mm": round(y, 3),
                "w_mm": PATCH_MM,
                "h_mm": PATCH_MM,
            }
        )
        px_ += PATCH_MM + PATCH_GAP_MM
    y += PATCH_MM + 10

    # ---- mounting note ---------------------------------------------------
    for line in [
        "MOUNTING",
        "  Trim on the crop marks and tape to a headband so the marker sits flat, fully in frame",
        "  and in the same focal plane as your cheek - not angled away, not behind your ear.",
        "  The marker must be visible in ALL THREE poses. Check the 60 deg poses specifically:",
        "  a headband marker on the left temple disappears in the right-60 shot.",
        "",
        "  Reprint whenever the sheet creases, yellows or picks up a shine. It is one page.",
    ]:
        d.text((mm(margin), mm(y)), line, font=f_sml, fill=(0, 0, 0))
        y += 5.0

    # ---- crop marks ------------------------------------------------------
    cm, ln = 8.0, 6.0
    for cx, cy, dx, dy in [
        (cm, cm, 1, 1), (PAGE_W_MM - cm, cm, -1, 1),
        (cm, PAGE_H_MM - cm, 1, -1), (PAGE_W_MM - cm, PAGE_H_MM - cm, -1, -1),
    ]:
        d.line([mm(cx), mm(cy), mm(cx + dx * ln), mm(cy)], fill=(0, 0, 0), width=max(1, mm(0.3)))
        d.line([mm(cx), mm(cy), mm(cx), mm(cy + dy * ln)], fill=(0, 0, 0), width=max(1, mm(0.3)))

    # ---- write -----------------------------------------------------------
    png = out_dir / "calibration_target.png"
    pdf = out_dir / "calibration_target.pdf"
    meta_p = out_dir / "calibration_target.json"

    img.save(png, dpi=(DPI, DPI))
    img.save(pdf, "PDF", resolution=DPI)

    meta = {
        "schema_version": 1,
        "page": {"w_mm": PAGE_W_MM, "h_mm": PAGE_H_MM, "px_per_mm": PX_PER_MM, "dpi": DPI},
        "marker": {
            "dict": config.ARUCO_DICT_NAME,
            "id": MARKER_ID,
            "side_mm": MARKER_MM,
            "side_px_on_sheet": side_px,
            "quiet_zone_mm": QUIET_MM,
            "x_mm": round(mk_x, 3),
            "y_mm": round(mk_y, 3),
            "measured_across": "outer black edge",
        },
        "ruler": {"length_mm": RULER_MM, "tick_mm": 10.0},
        "patches": patch_geom,
        "print_instructions": "100% scale, no fit-to-page, matte paper",
    }
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"wrote {png}  ({img.width}x{img.height} px @ {DPI:.1f} dpi)")
    print(f"wrote {pdf}")
    print(f"wrote {meta_p}")
    print(f"marker rendered at {side_px} px = {side_px / PX_PER_MM:.3f} mm (target {MARKER_MM} mm)")
    return meta


# --------------------------------------------------------------------------
# detection - used by verify_print, qa.py and register.py
# --------------------------------------------------------------------------

@dataclass
class FiducialResult:
    found: bool
    marker_id: int | None = None
    corners: list[list[float]] = field(default_factory=list)   # 4x2, clockwise from top-left
    edges_px: list[float] = field(default_factory=list)        # 4 edge lengths
    mean_edge_px: float = 0.0
    px_per_mm: float | None = None
    edge_spread_pct: float = 0.0        # max deviation from the mean edge
    diag_ratio: float = 0.0             # 1.0 for a square-on view
    opposite_edge_ratio: float = 0.0    # 1.0 for a square-on view
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _detector() -> "cv2.aruco.ArucoDetector":
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT_NAME))
    params = cv2.aruco.DetectorParameters()
    # Subpixel refinement matters: px/mm is the divisor on every millimetre we
    # report, so a half-pixel corner error propagates into every measurement.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 7
    params.cornerRefinementMaxIterations = 60
    params.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.ArucoDetector(adict, params)


def detect_fiducial(img_bgr: np.ndarray, marker_id: int = MARKER_ID) -> FiducialResult:
    """Find the 30 mm marker and derive px/mm for THIS image.

    Per CLAUDE.md the scale is always per-image. Never cache this across
    sessions: standing 5 cm closer changes it by 15%.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    corners, ids, _ = _detector().detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return FiducialResult(found=False, note="no ArUco markers detected")

    ids_flat = ids.ravel().tolist()
    if marker_id not in ids_flat:
        return FiducialResult(
            found=False, note=f"marker id {marker_id} not among detected ids {ids_flat}"
        )

    quad = corners[ids_flat.index(marker_id)].reshape(4, 2).astype(float)

    edges = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    mean_edge = float(np.mean(edges))
    spread = float((max(edges) - min(edges)) / mean_edge * 100.0) if mean_edge else 0.0

    d1 = float(np.linalg.norm(quad[2] - quad[0]))
    d2 = float(np.linalg.norm(quad[3] - quad[1]))
    diag_ratio = float(max(d1, d2) / min(d1, d2)) if min(d1, d2) > 0 else 0.0
    opp = float(max(edges[0], edges[2]) / min(edges[0], edges[2])) if min(edges[0], edges[2]) > 0 else 0.0

    return FiducialResult(
        found=True,
        marker_id=marker_id,
        corners=quad.tolist(),
        edges_px=edges,
        mean_edge_px=mean_edge,
        px_per_mm=mean_edge / MARKER_MM,
        edge_spread_pct=spread,
        diag_ratio=diag_ratio,
        opposite_edge_ratio=opp,
        note="ok",
    )


# --------------------------------------------------------------------------
# verify_print
# --------------------------------------------------------------------------

def verify_print(image_path: Path, tol_pct: float = 2.0) -> bool:
    """Check a photo of the printed sheet. Returns True on PASS.

    HONEST LIMITATION, stated up front: edge-length disagreement conflates
    two different faults - a print that scaled, and a camera that was not
    square-on to the paper. This function cannot separate them, because a
    marker photographed at an angle has unequal edges no matter how
    perfectly it printed.

    So this verifies DETECTION and SQUARE-ON CAPTURE. Absolute print scale
    is confirmed with a physical ruler against the 30 mm marker and the
    100 mm line. Do that once, with a ruler, and never worry about it again.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"FAIL - could not read {image_path}")
        return False

    r = detect_fiducial(img)
    print("=" * 66)
    print(f"verify_print  {image_path.name}   ({img.shape[1]}x{img.shape[0]} px)")
    print("=" * 66)

    if not r.found:
        print(f"FAIL - {r.note}")
        print("  Check: is the whole marker in frame, in focus, and not blown out?")
        return False

    print(f"  marker id            {r.marker_id}")
    print(f"  edge lengths (px)    " + ", ".join(f"{e:.1f}" for e in r.edges_px))
    print(f"  mean edge            {r.mean_edge_px:.2f} px  for a {MARKER_MM:.1f} mm side")
    print(f"  measured scale       {r.px_per_mm:.3f} px/mm")
    print(f"  edge spread          {r.edge_spread_pct:.2f} %   (tolerance {tol_pct:.1f} %)")
    print(f"  diagonal ratio       {r.diag_ratio:.4f}   (1.0000 = square-on)")
    print(f"  opposite-edge ratio  {r.opposite_edge_ratio:.4f}   (1.0000 = square-on)")

    lesion_mm = 1.5
    print(f"  -> a {lesion_mm} mm lesion spans {lesion_mm * r.px_per_mm:.1f} px at this scale")

    ok = r.edge_spread_pct <= tol_pct
    print("-" * 66)
    if ok:
        print("PASS - marker detected, four edges agree, capture was square-on.")
    else:
        print("FAIL - the four edges disagree by more than the tolerance.")
        print("  Most likely the camera was not square-on to the sheet. Re-shoot")
        print("  straight-on before concluding the print is wrong.")
    print()
    print("  STILL REQUIRED, and this script cannot do it for you:")
    print("  put a ruler on the printed marker. Outer black edge must read 30.0 mm,")
    print("  and the reference line must read 100.0 mm. If they do not, reprint with")
    print("  scaling off. Every millimetre in this project divides by that number.")
    return ok


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="render the printable calibration sheet")
    g.add_argument("--out", type=Path, default=None)

    v = sub.add_parser("verify", help="check a photograph of the printed sheet")
    v.add_argument("image", type=Path)
    v.add_argument("--tol", type=float, default=2.0, help="edge spread tolerance in %%")

    a = ap.parse_args(argv)
    if a.cmd == "generate":
        generate(a.out)
        return 0
    return 0 if verify_print(a.image, a.tol) else 1


if __name__ == "__main__":
    raise SystemExit(main())
