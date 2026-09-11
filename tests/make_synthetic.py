"""Synthesise test images so every phase 0 code path can be exercised today.

None of this is data. It exists only to prove the tooling does not crash and,
more usefully, to validate register.py against a transform we KNOW exactly.

Two products:

  sheet      a simulated photograph of the printed calibration sheet
             (perspective + blur + noise + vignette) for fiducial.verify_print

  faces      a reference face carrying an ArUco marker, plus N target images
             generated from it by KNOWN homographies, plus control-point
             files derived by pushing reference points through those same
             homographies.

The faces case is the important one: because the ground-truth warp is known,
the residuals register.py reports should be ~0 for a method that recovers it.
A method that reports 4 mm on synthetic data has a bug, not a hard problem.

    python -m tests.make_synthetic all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, fiducial  # noqa: E402

OUT = config.FIXTURES_DIR / "synthetic"
RNG = np.random.default_rng(20260911)


def _degrade(img: np.ndarray, blur: float = 1.1, noise: float = 3.0, vignette: float = 0.30) -> np.ndarray:
    """Make a clean render look like it came off a phone sensor."""
    out = img.astype(np.float32)
    if blur > 0:
        k = int(blur * 4) | 1
        out = cv2.GaussianBlur(out, (k, k), blur)
    if vignette > 0:
        h, w = out.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h / 2, w / 2
        r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
        out *= (1.0 - vignette * np.clip(r / 1.414, 0, 1) ** 2)[..., None]
    if noise > 0:
        out += RNG.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _perspective(w: int, h: int, jitter: float) -> np.ndarray:
    """Random mild perspective, as if the camera were slightly off-axis."""
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    d = jitter * min(w, h)
    dst = src + RNG.uniform(-d, d, src.shape).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst)


# --------------------------------------------------------------------------

def make_sheet() -> Path:
    """Simulated photo of the printed calibration target."""
    sheet_p = config.REPO / "docs" / "fiducial" / "calibration_target.png"
    if not sheet_p.exists():
        fiducial.generate()
    sheet = cv2.imread(str(sheet_p))

    # Simulate framing the sheet with a phone: downscale to sensor-ish size.
    sheet = cv2.resize(sheet, (1512, 1956), interpolation=cv2.INTER_AREA)
    h, w = sheet.shape[:2]
    canvas = np.full((2016, 1512, 3), 235, np.uint8)          # paper on a desk
    y0 = (canvas.shape[0] - h) // 2
    canvas[y0:y0 + h, 0:w] = sheet

    H = _perspective(canvas.shape[1], canvas.shape[0], 0.006)  # nearly square-on
    warped = cv2.warpPerspective(canvas, H, (canvas.shape[1], canvas.shape[0]),
                                 borderValue=(235, 235, 235))
    warped = _degrade(warped, blur=1.0, noise=2.5, vignette=0.22)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "sheet_photo.jpg"
    cv2.imwrite(str(p), warped, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {p}  {warped.shape[1]}x{warped.shape[0]}")
    return p


# --------------------------------------------------------------------------

def _paste_marker(img: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Composite a 30 mm ArUco marker with its quiet zone onto the frame."""
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT_NAME))
    side = int(round(config.ARUCO_SIDE_MM * px_per_mm))
    side = max(6, (side // 6) * 6)
    m = cv2.aruco.generateImageMarker(adict, config.ARUCO_ID, side)
    m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

    quiet = int(round(8 * px_per_mm))
    tile = np.full((side + 2 * quiet, side + 2 * quiet, 3), 255, np.uint8)
    tile[quiet:quiet + side, quiet:quiet + side] = m

    out = img.copy()
    # Top-left, as a headband marker would sit.
    y0, x0 = int(0.035 * out.shape[0]), int(0.045 * out.shape[1])
    th, tw = tile.shape[:2]
    if y0 + th < out.shape[0] and x0 + tw < out.shape[1]:
        out[y0:y0 + th, x0:x0 + tw] = tile
    return out


def make_faces(n_targets: int = 4) -> Path:
    """Reference + N targets related by KNOWN homographies, with control points."""
    src_p = config.FIXTURES_DIR / "portrait.jpg"
    if not src_p.exists():
        raise SystemExit(f"missing {src_p} - run scripts/fetch_weights.py first")

    face = cv2.imread(str(src_p))
    # Upscale so a 30 mm marker and 1.5 mm features are plausibly resolved.
    face = cv2.resize(face, None, fx=2.6, fy=2.6, interpolation=cv2.INTER_CUBIC)
    h, w = face.shape[:2]

    # Pick a scale that puts the marker at a believable size in frame.
    px_per_mm = w / 210.0                      # ~face+headroom across ~21 cm
    ref = _paste_marker(face, px_per_mm)
    ref = _degrade(ref, blur=0.8, noise=2.0, vignette=0.18)

    d = OUT / "faces"
    d.mkdir(parents=True, exist_ok=True)
    ref_p = d / "ref.jpg"
    cv2.imwrite(str(ref_p), ref, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Control points on the REFERENCE, spread across the face area. In the real
    # workflow these are hand-clicked moles; here they are arbitrary but fixed,
    # and we push them through the known warp to get each target's copy.
    cps = []
    for i in range(10):
        cx = w * (0.28 + 0.44 * ((i % 5) / 4.0))
        cy = h * (0.30 + 0.45 * ((i // 5) / 1.0))
        cps.append({"id": i, "x_px": float(cx), "y_px": float(cy)})

    # Deliberately NOT config.CONTROLPOINTS_DIR. These ids are throwaway, and
    # controlpoints.roster_ids() scans that whole directory - synthetic ids
    # leaking in would corrupt the roster for real moles.
    cpdir = OUT / "controlpoints"
    cpdir.mkdir(parents=True, exist_ok=True)
    (cpdir / "ref.json").write_text(
        json.dumps({"image": "ref.jpg", "points": cps}, indent=2), encoding="utf-8"
    )

    truth = {"reference": "ref.jpg", "px_per_mm_nominal": px_per_mm, "targets": {}}

    for t in range(n_targets):
        # A realistic day-to-day difference: small rotation, scale, shift, and
        # a touch of perspective from not standing in exactly the same spot.
        ang = RNG.uniform(-4.0, 4.0)
        sc = RNG.uniform(0.96, 1.04)
        tx, ty = RNG.uniform(-0.03, 0.03, 2) * [w, h]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        M[0, 2] += tx
        M[1, 2] += ty
        Hsim = np.vstack([M, [0, 0, 1]]).astype(np.float64)
        Hpersp = _perspective(w, h, 0.004)
        H = Hpersp @ Hsim

        warped = cv2.warpPerspective(face, H, (w, h), borderValue=(120, 120, 120))
        warped = _paste_marker(warped, px_per_mm)          # marker re-pasted flat, as in life
        warped = _degrade(warped, blur=RNG.uniform(0.7, 1.4), noise=RNG.uniform(1.5, 3.5),
                          vignette=RNG.uniform(0.14, 0.26))

        name = f"target_{t+1}.jpg"
        cv2.imwrite(str(d / name), warped, [cv2.IMWRITE_JPEG_QUALITY, 92])

        pts = np.array([[p["x_px"], p["y_px"]] for p in cps], np.float64).reshape(-1, 1, 2)
        moved = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        (cpdir / f"{Path(name).stem}.json").write_text(
            json.dumps(
                {
                    "image": name,
                    "points": [
                        {"id": int(c["id"]), "x_px": float(mx), "y_px": float(my)}
                        for c, (mx, my) in zip(cps, moved)
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        truth["targets"][name] = {"H_target_from_ref": H.tolist(), "rot_deg": ang, "scale": sc}
        print(f"wrote {d / name}")

    (d / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"wrote {ref_p}")
    print(f"wrote {d / 'ground_truth.json'}")
    print(f"wrote control points -> {cpdir}")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sheet", "faces", "all"], default="all", nargs="?")
    a = ap.parse_args()
    config.ensure_dirs()
    if a.what in ("sheet", "all"):
        make_sheet()
    if a.what in ("faces", "all"):
        make_faces()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
