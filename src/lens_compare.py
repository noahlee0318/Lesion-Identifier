"""Main lens @ ~30 cm vs 2x lens @ ~55 cm - phase 0 question (c).

The question is NOT "which looks sharper". A tighter crop of the same skin
looks sharper while resolving no more detail. The question is which lens
resolves usable detail at the sideburn without giving up the nose, and the
only scale-free way to answer that is CYCLES PER MILLIMETRE OF SKIN.

So every region is cropped to the same PHYSICAL area (default 14 mm square)
using that image's own ArUco px/mm, and detail is measured in cycles/mm. A
lens that magnifies more gets a bigger pixel crop of the same skin - which is
exactly the advantage we are trying to measure, and exactly what a
pixel-based sharpness number would hide.

ON MTF50. A slanted-edge MTF needs a straight high-contrast edge. Skin has
none, and forcing one (a hair, the marker border) measures a different depth
plane than the region under test. Instead this uses the radially-averaged
power spectrum of the crop, reporting the spatial frequency below which a
given fraction of the AC energy sits. It is honest about texture and needs no
edge. It is reported as `detail_cy_per_mm`, not as MTF50.

    python -m src.lens_compare --main SHOT_main.jpg --tele SHOT_2x.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    from . import config, facemesh_check, fiducial
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config, facemesh_check, fiducial

CROP_MM = 14.0          # physical size of every analysis crop
ENERGY_FRAC = 0.80      # detail limit = frequency holding this much AC energy

# Region anchors, per image side. The near side is chosen automatically.
REGIONS_BY_SIDE = {
    "nose_tip": {"both": [1, 4, 19, 94, 2]},
    "medial_cheek": {"img_l": [205, 50, 118], "img_r": [425, 280, 347]},
    "lateral_cheek": {"img_l": [116, 123, 117, 111], "img_r": [345, 352, 346, 340]},
    "sideburn_preauricular": {"img_l": [127, 162, 234, 93], "img_r": [356, 389, 454, 323]},
}


@dataclass
class RegionMeasure:
    region: str
    ok: bool = False
    cx: float = 0.0
    cy: float = 0.0
    crop_px: int = 0
    px_per_mm: float = 0.0
    lap_var: float = float("nan")
    detail_cy_per_mm: float = float("nan")
    detail_cy_per_px: float = float("nan")
    mean_luma: float = float("nan")
    clipped_frac: float = float("nan")
    note: str = ""


@dataclass
class LensMeasure:
    path: Path
    lens: str
    ok: bool = False
    px_per_mm: float = 0.0
    yaw: float = float("nan")
    near_side: str = ""
    regions: dict[str, RegionMeasure] = field(default_factory=dict)
    note: str = ""


# --------------------------------------------------------------------------

def _radial_detail(gray: np.ndarray, px_per_mm: float, frac: float = ENERGY_FRAC) -> tuple[float, float]:
    """Spatial frequency below which `frac` of the AC energy lies.

    Returns (cycles/mm, cycles/px). Higher means finer detail survived.
    """
    g = gray.astype(np.float32)
    g -= g.mean()
    if g.size < 64 or g.std() < 1e-6:
        return float("nan"), float("nan")

    # Hann window kills the spectral leakage that a hard crop edge would add.
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(g * win))
    P = (F.real ** 2 + F.imag ** 2)

    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    # Normalised radius in cycles/px: 0.5 at Nyquist.
    r = np.sqrt(((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2)

    nb = 64
    edges = np.linspace(0, 0.5, nb + 1)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, nb - 1)
    prof = np.bincount(idx, weights=P.ravel(), minlength=nb)
    prof[0] = 0.0                                  # drop DC
    tot = prof.sum()
    if tot <= 0:
        return float("nan"), float("nan")

    c = np.cumsum(prof) / tot
    k = int(np.searchsorted(c, frac))
    k = min(k, nb - 1)
    cy_per_px = float((edges[k] + edges[k + 1]) / 2.0)
    return cy_per_px * px_per_mm, cy_per_px


def measure_image(path: Path, lens: str, landmarker) -> LensMeasure:
    m = LensMeasure(path=path, lens=lens)
    img = cv2.imread(str(path))
    if img is None:
        m.note = "unreadable"
        return m

    fr = fiducial.detect_fiducial(img)
    if not fr.found or not fr.px_per_mm:
        m.note = f"no fiducial ({fr.note}) - cannot express detail per millimetre"
        return m
    m.px_per_mm = fr.px_per_mm

    r = facemesh_check.analyse(path, landmarker)
    if not r.ok or r.lm_px is None:
        m.note = f"Face Mesh failed ({r.note}) - cannot locate regions"
        return m
    m.yaw = r.yaw
    lm = r.lm_px

    # Which side faces the camera? Use the anchor visibility already computed.
    lat_l = [i for i in REGIONS_BY_SIDE["lateral_cheek"]["img_l"] if i < len(lm)]
    lat_r = [i for i in REGIONS_BY_SIDE["lateral_cheek"]["img_r"] if i < len(lm)]
    face_cx = lm[:, 0].mean()
    # The near side in a turned pose is the one with more horizontal spread
    # from the face centre - the far side compresses under foreshortening.
    spread_l = float(np.abs(lm[lat_l, 0] - face_cx).mean()) if lat_l else 0.0
    spread_r = float(np.abs(lm[lat_r, 0] - face_cx).mean()) if lat_r else 0.0
    m.near_side = "img_l" if spread_l >= spread_r else "img_r"

    half = int(round(CROP_MM * m.px_per_mm / 2.0))
    if half < 8:
        m.note = f"scale too low ({m.px_per_mm:.2f} px/mm) for a {CROP_MM} mm crop"
        return m

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    for region, sides in REGIONS_BY_SIDE.items():
        idxs = sides.get("both") or sides.get(m.near_side, [])
        idxs = [i for i in idxs if i < len(lm)]
        rm = RegionMeasure(region=region, px_per_mm=m.px_per_mm, crop_px=2 * half)
        if not idxs:
            rm.note = "no anchors"
            m.regions[region] = rm
            continue
        cx, cy = float(lm[idxs, 0].mean()), float(lm[idxs, 1].mean())
        rm.cx, rm.cy = cx, cy
        x0, y0 = int(round(cx)) - half, int(round(cy)) - half
        if x0 < 0 or y0 < 0 or x0 + 2 * half > W or y0 + 2 * half > H:
            rm.note = "crop falls outside the frame"
            m.regions[region] = rm
            continue
        crop = gray[y0:y0 + 2 * half, x0:x0 + 2 * half]
        rm.lap_var = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        rm.detail_cy_per_mm, rm.detail_cy_per_px = _radial_detail(crop, m.px_per_mm)
        rm.mean_luma = float(crop.mean())
        rm.clipped_frac = float(((crop >= 250) | (crop <= 5)).mean())
        rm.ok = True
        m.regions[region] = rm

    m.ok = True
    return m


# --------------------------------------------------------------------------

def montage(mains: LensMeasure, tele: LensMeasure, out: Path) -> Path | None:
    """Same physical skin, side by side, at each lens's native resolution."""
    regions = [r for r in REGIONS_BY_SIDE if
               mains.regions.get(r, RegionMeasure(r)).ok and tele.regions.get(r, RegionMeasure(r)).ok]
    if not regions:
        return None

    imgs = {"main": cv2.imread(str(mains.path)), "2x": cv2.imread(str(tele.path))}
    cell = 320
    rows = []
    for region in regions:
        row = []
        for lens, meas in (("main", mains), ("2x", tele)):
            rm = meas.regions[region]
            half = rm.crop_px // 2
            x0, y0 = int(round(rm.cx)) - half, int(round(rm.cy)) - half
            crop = imgs[lens][y0:y0 + rm.crop_px, x0:x0 + rm.crop_px]
            # Nearest-neighbour so the true pixel grid stays visible.
            crop = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_NEAREST)
            bar = np.zeros((54, cell, 3), np.uint8)
            for i, t in enumerate([
                f"{lens}  {region}",
                f"{rm.crop_px}px / {CROP_MM:.0f}mm = {rm.px_per_mm:.1f} px/mm",
                f"detail {rm.detail_cy_per_mm:.2f} cy/mm   lap {rm.lap_var:.0f}",
            ]):
                cv2.putText(bar, t, (5, 15 + i * 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.36, (255, 255, 255), 1, cv2.LINE_AA)
            row.append(np.vstack([crop, bar]))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out


def write_report(mains: LensMeasure, tele: LensMeasure, out: Path, mont: Path | None) -> None:
    L = ["# Lens comparison - phase 0 question (c)\n"]

    if not (mains.ok and tele.ok):
        L.append("## VERDICT\n\n**Could not compare.**\n")
        for m in (mains, tele):
            if not m.ok:
                L.append(f"- `{m.lens}` ({m.path.name}): {m.note}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(L), encoding="utf-8")
        print(f"wrote {out}")
        return

    def d(m: LensMeasure, r: str) -> float:
        rm = m.regions.get(r)
        return rm.detail_cy_per_mm if rm and rm.ok else float("nan")

    sideburn_main, sideburn_tele = d(mains, "sideburn_preauricular"), d(tele, "sideburn_preauricular")
    nose_main, nose_tele = d(mains, "nose_tip"), d(tele, "nose_tip")

    L.append("## VERDICT\n")
    if np.isfinite(sideburn_main) and np.isfinite(sideburn_tele):
        def verdict_line(label: str, v_main: float, v_tele: float) -> tuple[str, str]:
            # Within 3% is a tie, not a win - these are texture statistics on
            # one pair of shots, not a repeated measurement.
            if abs(v_tele - v_main) <= 0.03 * max(v_main, v_tele, 1e-9):
                win, head = "tie", "**level**"
            else:
                win = "2x" if v_tele > v_main else "main"
                head = f"`{win}` resolves more"
            return win, (f"- **{label}:** {head} "
                         f"(main {v_main:.2f} vs 2x {v_tele:.2f} cy/mm)")

        sb_win, sb_line = verdict_line("Sideburn / preauricular", sideburn_main, sideburn_tele)
        nose_win, nose_line = verdict_line("Nose tip", nose_main, nose_tele)
        L.append(sb_line)
        L.append(nose_line)
        if sb_win == "tie":
            L.append("\n**No measurable difference at the sideburn.** Pick the main lens for "
                     "working distance and depth of field, and re-run if you later suspect the "
                     "sideburn is under-resolved.\n")
        elif nose_win in (sb_win, "tie"):
            L.append(f"\n**Use the {sb_win} lens.** It wins at the region that motivated the "
                     "whole 60 deg design and does not give up the nose to do it.\n")
        else:
            loss = abs(nose_tele - nose_main) / max(nose_main, nose_tele, 1e-9) * 100
            L.append(f"\n**Trade-off: `{sb_win}` wins the sideburn, `{nose_win}` wins the nose.** "
                     f"The nose costs {loss:.0f}% of its detail. The sideburn is the region the "
                     "60 deg pose exists to capture and the nose is already well covered by the "
                     f"frontal pose, so take `{sb_win}` unless the nose loss exceeds ~30%.\n")
    else:
        L.append("\n**Inconclusive** - one or both sideburn crops could not be measured.\n")

    L.append(f"\nCrops are {CROP_MM:.0f} mm square of real skin in both images, located by Face "
             "Mesh so they land on the same anatomy, and sized through each image's own ArUco "
             "scale. `detail` is the spatial frequency holding "
             f"{ENERGY_FRAC:.0%} of the crop's AC energy.\n")

    L.append("\n## Per region\n")
    L.append("| region | lens | px/mm | crop px | detail cy/mm | laplacian var | mean luma | clipped |")
    L.append("|---|---|---|---|---|---|---|---|")
    for region in REGIONS_BY_SIDE:
        for m in (mains, tele):
            rm = m.regions.get(region)
            if not rm or not rm.ok:
                L.append(f"| {region} | {m.lens} | - | - | - | - | - | {rm.note if rm else 'missing'} |")
                continue
            L.append(
                f"| {region} | {m.lens} | {rm.px_per_mm:.2f} | {rm.crop_px} | "
                f"{rm.detail_cy_per_mm:.2f} | {rm.lap_var:.0f} | {rm.mean_luma:.0f} | "
                f"{rm.clipped_frac:.3f} |"
            )

    L.append("\n## Capture context\n")
    L.append("| lens | file | ArUco px/mm | yaw | near side |")
    L.append("|---|---|---|---|---|")
    for m in (mains, tele):
        L.append(f"| {m.lens} | {m.path.name} | {m.px_per_mm:.2f} | {m.yaw:+.1f} | {m.near_side} |")

    if mont:
        L.append(f"\n## Visual check\n\n`{mont.name}` - the same millimetres of skin from each "
                 "lens at native pixel density, nearest-neighbour upscaled so the real grid is "
                 "visible. Trust your eyes here over the table if they disagree.\n")

    L.append("\n## Caveats\n")
    L.append(f"- `px/mm` is measured at the ArUco marker's depth. The nose and the sideburn sit "
             "several centimetres in front of and behind it in a turned pose, so their true "
             "scale differs from the marker's by roughly that depth ratio. The lens-vs-lens "
             "comparison within a region is unaffected, because both images have the same "
             "geometry problem - but do not read the absolute cy/mm as exact.")
    L.append("- Laplacian variance is included because it is the conventional number, but it is "
             "resolution-dependent and will favour whichever lens magnifies more. Where it "
             "disagrees with cy/mm, cy/mm is the one that answers the question.")
    L.append("- One pose, one pair of shots. If the two were taken minutes apart with different "
             "focus locks, this measures focus luck as much as optics. Shoot the pair back to "
             "back and re-run if the numbers look close.\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="main vs 2x lens comparison")
    ap.add_argument("--main", type=Path, required=True, help="main-lens image (~30 cm)")
    ap.add_argument("--tele", type=Path, required=True, help="2x-lens image (~55 cm)")
    ap.add_argument("--report", type=Path, default=config.REPORTS_DIR / "lens_compare.md")
    ap.add_argument("--montage", type=Path, default=config.REPORTS_DIR / "lens_compare_crops.jpg")
    a = ap.parse_args(argv)

    landmarker = facemesh_check._make_landmarker()
    m = measure_image(a.main, "main", landmarker)
    t = measure_image(a.tele, "2x", landmarker)
    for x in (m, t):
        print(f"  {x.lens:5s} {x.path.name}  {'ok' if x.ok else 'FAILED: ' + x.note}"
              + (f"  {x.px_per_mm:.2f} px/mm  near={x.near_side}" if x.ok else ""))

    mont = montage(m, t, a.montage) if (m.ok and t.ok) else None
    if mont:
        print(f"wrote {mont}")
    write_report(m, t, a.report, mont)

    j = a.report.with_suffix(".json")
    j.write_text(
        json.dumps(
            {
                x.lens: {
                    "path": str(x.path), "ok": x.ok, "px_per_mm": x.px_per_mm,
                    "yaw": x.yaw, "near_side": x.near_side, "note": x.note,
                    "regions": {k: vars(v) for k, v in x.regions.items()},
                }
                for x in (m, t)
            },
            indent=2, default=float,
        ),
        encoding="utf-8",
    )
    print(f"wrote {j}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
