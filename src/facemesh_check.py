"""Does MediaPipe hold at 60 deg yaw? This script answers (a) from the spike.

For every image in a directory it reports:
  - detection success
  - head pose (yaw/pitch/roll), two independent ways
  - which of the six region anchors are visible vs self-occluded
  - per-landmark positional stability across the set, in px AND mm

Then writes reports/facemesh_check.md and an annotated overlay per input.

NOTE ON THE API. mediapipe 1.0 removed the legacy `mp.solutions.face_mesh`
module entirely, so `FaceMesh(refine_landmarks=True)` no longer exists. The
replacement is the Tasks API `FaceLandmarker`, and the face_landmarker task
bundle emits 478 landmarks - 468 face + 10 iris - which is exactly what
refine_landmarks=True used to produce. Same landmark indices, same topology.

    python -m src.facemesh_check data/raw/calib
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
    from . import config, fiducial
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config, fiducial

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# --------------------------------------------------------------------------
# Landmark indices (canonical MediaPipe face mesh numbering)
# --------------------------------------------------------------------------

# SIDEDNESS ASSUMPTION - confirm this on the first real photo.
# The iPhone rear camera writes an UNMIRRORED file, so the subject's right
# side appears on the image left (smaller x). Anchor sets below are named
# anatomically (subject's left/right). If the first real left-60 shot shows
# the wrong set occluded, flip SUBJECT_RIGHT_IS_IMAGE_LEFT and re-run.
SUBJECT_RIGHT_IS_IMAGE_LEFT = True

NOSE_TIP = 1
CHIN = 152
NASION = 168           # bridge, between the eyes
EYE_INNER_R, EYE_INNER_L = 133, 362
EYE_OUTER_R, EYE_OUTER_L = 33, 263
MOUTH_R, MOUTH_L = 61, 291

# The alignment basis: rigid mid-face structure that does not move with
# expression. Used ONLY to remove global pose before measuring stability.
STABLE_SUBSET = [EYE_INNER_R, EYE_INNER_L, NASION, 6, 197, 195]

# Anchors per region, by MediaPipe index. "_img_l"/"_img_r" suffixes are
# resolved to anatomical sides by SUBJECT_RIGHT_IS_IMAGE_LEFT.
_REGION_ANCHORS_IMG = {
    "forehead": [10, 67, 297, 109, 338, 151],
    "chin_perioral": [152, 175, 18, 200, 199, 164],
    "cheek_img_l": [234, 227, 116, 123, 50, 205],
    "cheek_img_r": [454, 447, 345, 352, 280, 425],
    "temple_sideburn_img_l": [127, 162, 21, 54, 103],
    "temple_sideburn_img_r": [356, 389, 251, 284, 332],
}


def region_anchors() -> dict[str, list[int]]:
    """Map image-side anchor groups onto anatomical region names."""
    a = dict(_REGION_ANCHORS_IMG)
    if SUBJECT_RIGHT_IS_IMAGE_LEFT:
        img_l_side, img_r_side = "r", "l"      # image-left == subject's right
    else:
        img_l_side, img_r_side = "l", "r"
    return {
        "forehead": a["forehead"],
        "chin_perioral": a["chin_perioral"],
        f"cheek_{img_l_side}": a["cheek_img_l"],
        f"cheek_{img_r_side}": a["cheek_img_r"],
        f"temple_sideburn_{img_l_side}": a["temple_sideburn_img_l"],
        f"temple_sideburn_{img_r_side}": a["temple_sideburn_img_r"],
    }


# Approximate anthropometric head model in mm, for the solvePnP cross-check.
# These are the standard generic values used for head-pose estimation; they
# are NOT the MediaPipe canonical mesh. Treat this as a second opinion on the
# transformation-matrix pose, not as an independent ground truth.
PNP_MODEL_MM = np.array(
    [
        [0.0, 0.0, 0.0],           # nose tip
        [0.0, -63.6, -12.5],       # chin
        [-43.3, 32.7, -26.0],      # eye, subject's right outer
        [43.3, 32.7, -26.0],       # eye, subject's left outer
        [-28.9, -28.9, -24.1],     # mouth, subject's right
        [28.9, -28.9, -24.1],      # mouth, subject's left
    ],
    dtype=np.float64,
)
PNP_IDX = [NOSE_TIP, CHIN, EYE_OUTER_R, EYE_OUTER_L, MOUTH_R, MOUTH_L]


# --------------------------------------------------------------------------

@dataclass
class FrameResult:
    path: Path
    ok: bool
    w: int = 0
    h: int = 0
    n_landmarks: int = 0
    yaw: float = float("nan")
    pitch: float = float("nan")
    roll: float = float("nan")
    yaw_pnp: float = float("nan")
    pitch_pnp: float = float("nan")
    roll_pnp: float = float("nan")
    px_per_mm: float | None = None
    fiducial_note: str = ""
    low_vis: int = 0
    occluded_regions: dict[str, dict] = field(default_factory=dict)
    lm_px: np.ndarray | None = None      # (N,2) image pixels
    lm_xyz: np.ndarray | None = None     # (N,3) normalized, z is relative depth
    note: str = ""


# --------------------------------------------------------------------------

def _make_landmarker():
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not config.FACE_LANDMARKER_TASK.exists():
        raise SystemExit(
            f"missing {config.FACE_LANDMARKER_TASK}\nrun: python scripts/fetch_weights.py"
        )
    base = mp_python.BaseOptions(model_asset_path=str(config.FACE_LANDMARKER_TASK))
    opts = vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        # The transformation matrix is MediaPipe's own fit of the canonical
        # face model to the image. It is the better pose estimate of the two.
        output_facial_transformation_matrixes=True,
        output_face_blendshapes=False,
        min_face_detection_confidence=0.3,   # deliberately permissive: we are
        min_face_presence_confidence=0.3,    # testing whether it fires at all
        min_tracking_confidence=0.3,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def _euler_from_matrix(M: np.ndarray) -> tuple[float, float, float]:
    """Yaw/pitch/roll in degrees from a 4x4 (or 3x3) rotation.

    Sign convention chosen so a head turned to the subject's left reads
    negative yaw and to the subject's right reads positive, matching how the
    poses are named (left60 / right60).
    """
    R = M[:3, :3]
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return float(np.degrees(y)), float(np.degrees(x)), float(np.degrees(z))


def _solvepnp_pose(lm_px: np.ndarray, w: int, h: int) -> tuple[float, float, float]:
    """Second-opinion head pose from a generic 6-point model."""
    img_pts = lm_px[PNP_IDX].astype(np.float64)
    f = float(w)                                  # crude focal guess, ~60 deg HFOV
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float64)
    ok, rvec, _ = cv2.solvePnP(
        PNP_MODEL_MM, img_pts, K, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return (float("nan"),) * 3
    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = _euler_from_matrix(R)
    return yaw, pitch, roll


def _local_normals(xyz: np.ndarray, k: int = 12) -> np.ndarray:
    """Per-landmark surface normal from a PCA plane over k nearest neighbours.

    Used to decide self-occlusion. MediaPipe happily predicts landmarks on the
    far side of a turned head - it does not tell you they are invisible - so
    the only honest occlusion test is geometric: does the local surface face
    the camera or away from it?
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(xyz)
    centroid = xyz.mean(axis=0)
    normals = np.zeros_like(xyz)
    _, idx = tree.query(xyz, k=min(k, len(xyz)))
    for i, nb in enumerate(idx):
        P = xyz[nb] - xyz[nb].mean(axis=0)
        # Smallest singular vector is the plane normal.
        n = np.linalg.svd(P, full_matrices=False)[2][-1]
        # Orient outward from the head centre.
        if np.dot(n, xyz[i] - centroid) < 0:
            n = -n
        normals[i] = n
    return normals


def analyse(path: Path, landmarker) -> FrameResult:
    import mediapipe as mp

    img = cv2.imread(str(path))
    if img is None:
        return FrameResult(path=path, ok=False, note="unreadable image")
    h, w = img.shape[:2]

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res = landmarker.detect(mp_img)

    if not res.face_landmarks:
        return FrameResult(path=path, ok=False, w=w, h=h, note="NO FACE DETECTED")

    lms = res.face_landmarks[0]
    lm_xyz = np.array([[p.x, p.y, p.z] for p in lms], np.float64)
    lm_px = np.column_stack([lm_xyz[:, 0] * w, lm_xyz[:, 1] * h])

    # visibility/presence are frequently all-zero in the face task bundle;
    # count them only if the model actually populated them.
    vis = np.array([getattr(p, "visibility", 0.0) or 0.0 for p in lms])
    low_vis = int((vis < 0.5).sum()) if float(vis.max()) > 0 else -1

    r = FrameResult(path=path, ok=True, w=w, h=h, n_landmarks=len(lms), low_vis=low_vis)
    r.lm_px, r.lm_xyz = lm_px, lm_xyz

    if res.facial_transformation_matrixes:
        M = np.array(res.facial_transformation_matrixes[0]).reshape(4, 4)
        r.yaw, r.pitch, r.roll = _euler_from_matrix(M)
    r.yaw_pnp, r.pitch_pnp, r.roll_pnp = _solvepnp_pose(lm_px, w, h)

    # --- self-occlusion per region anchor ---------------------------------
    # Work in a camera-ish frame: x right, y down, z toward camera (MediaPipe
    # z is negative toward the camera), so the view direction is -z.
    metric = lm_xyz.copy()
    metric[:, 0] *= w
    metric[:, 1] *= h
    metric[:, 2] *= w          # z is normalised by image width
    normals = _local_normals(metric)
    view = np.array([0.0, 0.0, -1.0])
    facing = normals @ view     # >0 means the surface points at the camera

    for name, idxs in region_anchors().items():
        idxs = [i for i in idxs if i < len(lms)]
        f = facing[idxs]
        vis_frac = float((f > 0).mean()) if len(f) else 0.0
        r.occluded_regions[name] = {
            "anchors": len(idxs),
            "visible": int((f > 0).sum()),
            "visible_frac": round(vis_frac, 3),
            "status": "visible" if vis_frac >= 0.6 else ("partial" if vis_frac >= 0.3 else "OCCLUDED"),
            "mean_facing": round(float(f.mean()), 3) if len(f) else 0.0,
        }

    fr = fiducial.detect_fiducial(img)
    r.px_per_mm = fr.px_per_mm
    r.fiducial_note = fr.note
    return r


# --------------------------------------------------------------------------

def _similarity_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """2x3 similarity taking src -> dst, fitted on the stable subset only."""
    M, _ = cv2.estimateAffinePartial2D(
        src.reshape(-1, 1, 2).astype(np.float32),
        dst.reshape(-1, 1, 2).astype(np.float32),
        method=cv2.LMEDS,
    )
    return M


def stability(results: list[FrameResult]) -> dict:
    """Per-landmark scatter across the set, after removing global pose."""
    good = [r for r in results if r.ok and r.lm_px is not None]
    if len(good) < 2:
        return {"n": len(good), "note": "need >= 2 successful detections"}

    ref = good[0]
    aligned = [ref.lm_px]
    for r in good[1:]:
        M = _similarity_align(r.lm_px[STABLE_SUBSET], ref.lm_px[STABLE_SUBSET])
        if M is None:
            continue
        pts = cv2.transform(r.lm_px.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)
        aligned.append(pts.astype(np.float64))

    A = np.stack(aligned)                       # (K, N, 2)
    per_lm_std_px = A.std(axis=0).mean(axis=1)  # mean of x,y std per landmark

    scales = [r.px_per_mm for r in good if r.px_per_mm]
    px_per_mm = float(np.mean(scales)) if scales else None

    out = {
        "n_frames": len(A),
        "px_per_mm_mean": px_per_mm,
        "all_landmarks": {
            "median_std_px": float(np.median(per_lm_std_px)),
            "p95_std_px": float(np.percentile(per_lm_std_px, 95)),
            "max_std_px": float(per_lm_std_px.max()),
        },
        "per_region": {},
        "alignment_basis_is_circular": STABLE_SUBSET,
    }
    if px_per_mm:
        out["all_landmarks"].update(
            median_std_mm=float(np.median(per_lm_std_px) / px_per_mm),
            p95_std_mm=float(np.percentile(per_lm_std_px, 95) / px_per_mm),
            max_std_mm=float(per_lm_std_px.max() / px_per_mm),
        )

    for name, idxs in region_anchors().items():
        idxs = [i for i in idxs if i < len(per_lm_std_px)]
        s = per_lm_std_px[idxs]
        e = {"mean_std_px": float(s.mean()), "max_std_px": float(s.max())}
        if px_per_mm:
            e["mean_std_mm"] = float(s.mean() / px_per_mm)
            e["max_std_mm"] = float(s.max() / px_per_mm)
        out["per_region"][name] = e

    s = per_lm_std_px[STABLE_SUBSET]
    out["stable_subset"] = {
        "mean_std_px": float(s.mean()),
        "note": "this subset IS the alignment basis - its scatter is not an independent measure",
    }
    return out


# --------------------------------------------------------------------------

def draw_overlay(r: FrameResult, out_dir: Path) -> Path | None:
    if not r.ok or r.lm_px is None:
        return None
    img = cv2.imread(str(r.path))
    if img is None:
        return None
    vis = img.copy()

    for x, y in r.lm_px:
        cv2.circle(vis, (int(x), int(y)), 1, (0, 200, 200), -1, cv2.LINE_AA)

    colours = {"visible": (0, 200, 0), "partial": (0, 190, 255), "OCCLUDED": (0, 0, 255)}
    anchors = region_anchors()
    for name, info in r.occluded_regions.items():
        c = colours[info["status"]]
        for i in anchors[name]:
            if i < len(r.lm_px):
                x, y = r.lm_px[i]
                cv2.circle(vis, (int(x), int(y)), 7, c, 2, cv2.LINE_AA)

    scale = max(1.0, vis.shape[1] / 1600.0)
    lines = [
        f"{r.path.name}",
        f"yaw {r.yaw:+.1f}  pitch {r.pitch:+.1f}  roll {r.roll:+.1f}  (canonical fit)",
        f"yaw {r.yaw_pnp:+.1f}  pitch {r.pitch_pnp:+.1f}  roll {r.roll_pnp:+.1f}  (solvePnP)",
        f"landmarks {r.n_landmarks}   px/mm {r.px_per_mm:.2f}" if r.px_per_mm
        else f"landmarks {r.n_landmarks}   px/mm --  ({r.fiducial_note})",
    ]
    for i, t in enumerate(lines):
        y = int((24 + i * 26) * scale)
        cv2.putText(vis, t, (int(12 * scale), y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * scale, (0, 0, 0), int(4 * scale), cv2.LINE_AA)
        cv2.putText(vis, t, (int(12 * scale), y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * scale, (255, 255, 255), int(1 * scale), cv2.LINE_AA)

    y0 = int(140 * scale)
    for name, info in sorted(r.occluded_regions.items()):
        t = f"{name:24s} {info['status']:9s} {info['visible']}/{info['anchors']}"
        cv2.putText(vis, t, (int(12 * scale), y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale, (0, 0, 0), int(4 * scale), cv2.LINE_AA)
        cv2.putText(vis, t, (int(12 * scale), y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale, colours[info["status"]], int(1 * scale), cv2.LINE_AA)
        y0 += int(22 * scale)

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{r.path.stem}_overlay.jpg"
    cv2.imwrite(str(p), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return p


# --------------------------------------------------------------------------

def write_report(results: list[FrameResult], stab: dict, out: Path, target_yaw: float) -> None:
    n = len(results)
    ok = [r for r in results if r.ok]
    det = f"{len(ok)}/{n}"

    yaws = [abs(r.yaw) for r in ok if np.isfinite(r.yaw)]
    yaw_mean = float(np.mean(yaws)) if yaws else float("nan")
    yaw_sd = float(np.std(yaws)) if yaws else float("nan")

    cheek_temple = [
        v for k, v in stab.get("per_region", {}).items()
        if k.startswith("cheek") or k.startswith("temple")
    ]
    worst_mm = max((v.get("max_std_mm", float("nan")) for v in cheek_temple), default=float("nan"))

    detect_pass = len(ok) == n and n > 0
    stable_pass = np.isfinite(worst_mm) and worst_mm < 3.0

    L: list[str] = []
    L.append("# Face Mesh check - phase 0 question (a)\n")
    L.append("## VERDICT\n")
    L.append(f"- **Detection:** {det} images{'  PASS' if detect_pass else '  FAIL'}")
    if np.isfinite(worst_mm):
        L.append(
            f"- **Cheek/temple anchor stability:** worst {worst_mm:.2f} mm"
            f"{'  PASS (< 3 mm)' if stable_pass else '  FAIL (>= 3 mm)'}"
        )
    else:
        L.append("- **Cheek/temple anchor stability:** not measurable - no fiducial, so no mm scale")
    if yaws:
        L.append(f"- **Measured yaw:** {yaw_mean:.1f} deg mean, {yaw_sd:.1f} deg sd (target {target_yaw:.0f} deg)")

    if detect_pass and stable_pass:
        L.append(f"\n**HOLD AT {target_yaw:.0f} DEG.** Face Mesh fires on every frame and the "
                 "cheek/temple anchors are stable enough to anchor coarse alignment.\n")
    elif not detect_pass:
        L.append(f"\n**STEP DOWN TO 50 DEG.** Face Mesh did not fire on every frame at "
                 f"{target_yaw:.0f} deg. That is the documented fallback; re-shoot and re-run.\n")
    else:
        L.append("\n**MARGINAL.** Detection is fine but the anchors that matter most scatter "
                 "more than 3 mm. Before dropping the angle, check the rig: unstable anchors "
                 "usually mean the pose itself moved, not that the model failed.\n")

    L.append("\n## Per image\n")
    L.append("| image | detected | landmarks | yaw | pitch | roll | yaw (PnP) | px/mm |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        if not r.ok:
            L.append(f"| {r.path.name} | **NO** | - | - | - | - | - | - |")
            continue
        s = f"{r.px_per_mm:.2f}" if r.px_per_mm else "-"
        L.append(
            f"| {r.path.name} | yes | {r.n_landmarks} | {r.yaw:+.1f} | {r.pitch:+.1f} | "
            f"{r.roll:+.1f} | {r.yaw_pnp:+.1f} | {s} |"
        )

    L.append("\n`yaw` is MediaPipe's own fit of the canonical face model. `yaw (PnP)` is an "
             "independent solvePnP against a generic 6-point head model - a cross-check, not a "
             "second ground truth. Large disagreement means distrust both.\n")

    L.append("\n## Region anchor visibility\n")
    names = sorted(region_anchors().keys())
    L.append("| image | " + " | ".join(names) + " |")
    L.append("|---" * (len(names) + 1) + "|")
    for r in results:
        if not r.ok:
            L.append(f"| {r.path.name} |" + " - |" * len(names))
            continue
        cells = [r.occluded_regions.get(nm, {}).get("status", "?") for nm in names]
        L.append(f"| {r.path.name} | " + " | ".join(cells) + " |")
    L.append("\nSelf-occlusion is decided geometrically: a local surface normal fitted to each "
             "anchor's neighbourhood, tested against the view direction. MediaPipe predicts "
             "landmarks on the hidden side of a turned head without flagging them, so the "
             "model's own output cannot answer this.\n")

    L.append("\n## Landmark stability\n")
    if "note" in stab:
        L.append(f"_{stab['note']}_\n")
    else:
        a = stab["all_landmarks"]
        L.append(f"Across {stab['n_frames']} frames, after a similarity alignment on the stable "
                 f"mid-face subset {stab['alignment_basis_is_circular']}.\n")
        L.append("| scope | median | p95 | max |")
        L.append("|---|---|---|---|")
        if "median_std_mm" in a:
            L.append(f"| all 478 landmarks (mm) | {a['median_std_mm']:.2f} | "
                     f"{a['p95_std_mm']:.2f} | {a['max_std_mm']:.2f} |")
        L.append(f"| all 478 landmarks (px) | {a['median_std_px']:.2f} | "
                 f"{a['p95_std_px']:.2f} | {a['max_std_px']:.2f} |")
        L.append("\n| region | mean std | max std |")
        L.append("|---|---|---|")
        for nm, v in sorted(stab["per_region"].items()):
            if "mean_std_mm" in v:
                L.append(f"| {nm} | {v['mean_std_mm']:.2f} mm | {v['max_std_mm']:.2f} mm |")
            else:
                L.append(f"| {nm} | {v['mean_std_px']:.2f} px | {v['max_std_px']:.2f} px |")
        L.append(f"\n_{stab['stable_subset']['note']}._\n")

    L.append("\n## What this does not tell you\n")
    L.append("- Landmark stability is not registration accuracy. It says the coarse alignment "
             "has something reliable to start from; question (b) measures whether the fine fit "
             "actually lands.\n")
    L.append(f"- Sidedness assumption in force: subject's right appears on image left "
             f"(`SUBJECT_RIGHT_IS_IMAGE_LEFT = {SUBJECT_RIGHT_IS_IMAGE_LEFT}`). Confirm against "
             "the first real left-60 frame; if the wrong side reads OCCLUDED, flip it.\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MediaPipe Face Mesh reliability check")
    ap.add_argument("indir", type=Path, nargs="?", default=config.CALIB_DIR)
    ap.add_argument("--report", type=Path, default=config.REPORTS_DIR / "facemesh_check.md")
    ap.add_argument("--overlays", type=Path, default=config.REPORTS_DIR / "facemesh_overlays")
    ap.add_argument("--target-yaw", type=float, default=60.0)
    a = ap.parse_args(argv)

    if not a.indir.exists():
        print(f"no such directory: {a.indir}")
        return 2
    paths = sorted(p for p in a.indir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not paths:
        print(f"no images in {a.indir}")
        return 2

    print(f"landmarker: {config.FACE_LANDMARKER_TASK.name}")
    landmarker = _make_landmarker()
    results = []
    for p in paths:
        r = analyse(p, landmarker)
        results.append(r)
        if r.ok:
            print(f"  [ok]   {p.name}  yaw {r.yaw:+6.1f}  lm {r.n_landmarks}  "
                  f"px/mm {r.px_per_mm:.2f}" if r.px_per_mm else
                  f"  [ok]   {p.name}  yaw {r.yaw:+6.1f}  lm {r.n_landmarks}  px/mm --")
            draw_overlay(r, a.overlays)
        else:
            print(f"  [FAIL] {p.name}  {r.note}")

    stab = stability(results)
    write_report(results, stab, a.report, a.target_yaw)

    j = a.report.with_suffix(".json")
    j.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "image": r.path.name, "ok": r.ok, "yaw": r.yaw, "pitch": r.pitch,
                        "roll": r.roll, "yaw_pnp": r.yaw_pnp, "px_per_mm": r.px_per_mm,
                        "n_landmarks": r.n_landmarks, "regions": r.occluded_regions,
                        "note": r.note,
                    }
                    for r in results
                ],
                "stability": stab,
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(f"wrote {j}")
    print(f"overlays -> {a.overlays}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
