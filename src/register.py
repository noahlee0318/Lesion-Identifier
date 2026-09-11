"""Registration bake-off - phase 0 question (b), and the GO/NO-GO gate.

Runs a ladder of registration methods on the same reference/target pairs and
scores every one on the same metric: reprojection error at hand-marked
control points, in MILLIMETRES, converted through the reference image's own
ArUco px/mm.

  method 0  Face Mesh landmarks -> similarity                (coarse baseline)
  method 1  method 0, then SIFT + RANSAC -> homography
  method 2  method 0, then ORB + RANSAC -> homography
  method 3  method 0, then LightGlue + DISK -> homography
  method 4  LoFTR -> homography                              (detector-free)
  method 5  best of the above, then thin-plate spline on its matched points

GATE: median < 1.5 mm and p95 < 3 mm.

=============================================================================
CONTROL POINTS ARE HELD-OUT GROUND TRUTH. They are loaded in exactly one
place (`evaluate`) and are never passed to any transform estimator. If you
are about to fit something using them, stop.
=============================================================================

    python -m src.register bakeoff --ref REF.jpg --targets DIR_OR_FILES...
    python -m src.register review  --ref REF.jpg --target T.jpg --method tps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    from . import config, facemesh_check, fiducial
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config, facemesh_check, fiducial

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Resolution caps for the learned matchers. LoFTR's attention is quadratic in
# pixel count and the laptop 4050 has 6 GiB; matches are computed at the
# capped size and scaled back to full resolution, which costs a little
# localisation precision but is the difference between running and OOM.
LOFTR_MAX_SIDE = 1024
DISK_MAX_SIDE = 1600
SIFT_MAX_SIDE = 2400


# --------------------------------------------------------------------------
# control points - ground truth, read-only
# --------------------------------------------------------------------------

# Where control points are read from. Overridable so the synthetic fixtures
# never mix their throwaway ids into the real roster.
CP_DIR = config.CONTROLPOINTS_DIR


def set_controlpoints_dir(d: Path) -> None:
    global CP_DIR
    CP_DIR = Path(d)


def load_controlpoints(image_path: Path) -> dict[int, tuple[float, float]]:
    """Load {id: (x_px, y_px)} for one image, or {} if not marked yet."""
    p = CP_DIR / f"{image_path.stem}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {int(q["id"]): (float(q["x_px"]), float(q["y_px"])) for q in d.get("points", [])}


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------

class Transform:
    """Maps TARGET image coordinates into REFERENCE image coordinates."""

    kind = "none"

    def map_points(self, pts: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def warp_to_reference(self, img: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
        raise NotImplementedError


class HomographyTransform(Transform):
    def __init__(self, H: np.ndarray, kind: str = "homography"):
        self.H = np.asarray(H, np.float64)
        self.kind = kind

    def map_points(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def warp_to_reference(self, img, out_hw):
        h, w = out_hw
        return cv2.warpPerspective(img, self.H, (w, h), flags=cv2.INTER_LINEAR)


def _tps_kernel(r2: np.ndarray) -> np.ndarray:
    """U(r) = r^2 log(r^2), guarded at r = 0."""
    out = np.zeros_like(r2)
    nz = r2 > 1e-12
    out[nz] = r2[nz] * np.log(r2[nz])
    return out


class TPSTransform(Transform):
    """Thin-plate spline fitted on MATCHED FEATURE POINTS (never control points).

    Implemented directly rather than via cv2's shape transformer: the OpenCV
    API's source/destination argument order is ambiguous enough that getting
    it backwards produces plausible-looking but wrong numbers, and the
    regularisation parameter is not exposed in comparable units.

    Coordinates are normalised by the image diagonal before solving, which
    keeps the (n+3)x(n+3) system well conditioned at phone resolutions.
    """

    kind = "tps"

    def __init__(self, src: np.ndarray, dst: np.ndarray, lam: float = 1e-3, scale: float = 1.0):
        self.scale = float(scale)
        self.src = np.asarray(src, np.float64) / self.scale
        dstn = np.asarray(dst, np.float64) / self.scale
        n = len(self.src)

        diff = self.src[:, None, :] - self.src[None, :, :]
        K = _tps_kernel((diff ** 2).sum(-1))
        K += lam * np.eye(n)                      # regularise: smooth, not interpolating
        P = np.hstack([np.ones((n, 1)), self.src])

        L = np.zeros((n + 3, n + 3))
        L[:n, :n], L[:n, n:], L[n:, :n] = K, P, P.T
        Y = np.zeros((n + 3, 2))
        Y[:n] = dstn
        self.W = np.linalg.solve(L, Y)

    def map_points(self, pts: np.ndarray) -> np.ndarray:
        q = np.asarray(pts, np.float64).reshape(-1, 2) / self.scale
        diff = q[:, None, :] - self.src[None, :, :]
        U = _tps_kernel((diff ** 2).sum(-1))
        P = np.hstack([np.ones((len(q), 1)), q])
        out = U @ self.W[: len(self.src)] + P @ self.W[len(self.src):]
        return out * self.scale

    def warp_to_reference(self, img, out_hw):
        # Dense remap needs the INVERSE map (ref pixel -> target pixel), so the
        # caller supplies an inverse-fitted twin; see build_tps_pair.
        raise NotImplementedError("use TPSPair.warp_to_reference")


class TPSPair(Transform):
    """Forward TPS for points, inverse TPS for image warping."""

    kind = "tps"

    def __init__(self, fwd: TPSTransform, inv: TPSTransform):
        self.fwd, self.inv = fwd, inv

    def map_points(self, pts):
        return self.fwd.map_points(pts)

    def warp_to_reference(self, img, out_hw):
        h, w = out_hw
        step = max(1, int(min(h, w) / 240))       # solve on a coarse grid, upsample
        ys = np.arange(0, h, step, dtype=np.float64)
        xs = np.arange(0, w, step, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        mapped = self.inv.map_points(grid)
        mx = mapped[:, 0].reshape(gy.shape).astype(np.float32)
        my = mapped[:, 1].reshape(gy.shape).astype(np.float32)
        mx = cv2.resize(mx, (w, h), interpolation=cv2.INTER_LINEAR)
        my = cv2.resize(my, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# --------------------------------------------------------------------------
# matchers - each returns (pts_target, pts_reference) in FULL-RES pixels
# --------------------------------------------------------------------------

def _resize_cap(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(1.0, max_side / float(max(h, w)))
    if s >= 1.0:
        return img, 1.0
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA), s


def match_sift(tgt: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, sa = _resize_cap(cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY), SIFT_MAX_SIDE)
    b, sb = _resize_cap(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), SIFT_MAX_SIDE)
    sift = cv2.SIFT_create(nfeatures=8000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return np.empty((0, 2)), np.empty((0, 2))
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
    if not good:
        return np.empty((0, 2)), np.empty((0, 2))
    pa = np.float64([ka[m.queryIdx].pt for m in good]) / sa
    pb = np.float64([kb[m.trainIdx].pt for m in good]) / sb
    return pa, pb


def match_orb(tgt: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, sa = _resize_cap(cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY), SIFT_MAX_SIDE)
    b, sb = _resize_cap(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), SIFT_MAX_SIDE)
    orb = cv2.ORB_create(nfeatures=10000)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return np.empty((0, 2)), np.empty((0, 2))
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.8 * n.distance]
    if not good:
        return np.empty((0, 2)), np.empty((0, 2))
    pa = np.float64([ka[m.queryIdx].pt for m in good]) / sa
    pb = np.float64([kb[m.trainIdx].pt for m in good]) / sb
    return pa, pb


_DISK = _LG = _LOFTR = None


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def match_disk_lightglue(tgt: np.ndarray, ref: np.ndarray, n_feats: int = 4096):
    global _DISK, _LG
    import torch
    from kornia.feature import DISK, LightGlueMatcher, laf_from_center_scale_ori

    config.torch_home()
    dev = _device()
    if _DISK is None:
        _DISK = DISK.from_pretrained("depth").to(dev).eval()
        _LG = LightGlueMatcher("disk").to(dev).eval()

    def prep(img):
        small, s = _resize_cap(img, DISK_MAX_SIDE)
        t = torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        return t[None].to(dev), s, small.shape[:2]

    ta, sa, hwa = prep(tgt)
    tb, sb, hwb = prep(ref)
    with torch.inference_mode():
        fa = _DISK(ta, n=n_feats, window_size=5, score_threshold=0.0, pad_if_not_divisible=True)[0]
        fb = _DISK(tb, n=n_feats, window_size=5, score_threshold=0.0, pad_if_not_divisible=True)[0]
        if len(fa.keypoints) < 4 or len(fb.keypoints) < 4:
            return np.empty((0, 2)), np.empty((0, 2))
        la = laf_from_center_scale_ori(
            fa.keypoints[None],
            torch.ones(1, len(fa.keypoints), 1, 1, device=dev),
            torch.zeros(1, len(fa.keypoints), 1, device=dev),
        )
        lb = laf_from_center_scale_ori(
            fb.keypoints[None],
            torch.ones(1, len(fb.keypoints), 1, 1, device=dev),
            torch.zeros(1, len(fb.keypoints), 1, device=dev),
        )
        _, idxs = _LG(fa.descriptors, fb.descriptors, la, lb, hw1=hwa, hw2=hwb)
    if idxs.numel() == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    i = idxs.cpu().numpy()
    pa = fa.keypoints.cpu().numpy()[i[:, 0]].astype(np.float64) / sa
    pb = fb.keypoints.cpu().numpy()[i[:, 1]].astype(np.float64) / sb
    return pa, pb


def match_loftr(tgt: np.ndarray, ref: np.ndarray, conf_thr: float = 0.5):
    global _LOFTR
    import torch
    from kornia.feature import LoFTR

    config.torch_home()
    dev = _device()
    if _LOFTR is None:
        _LOFTR = LoFTR(pretrained="outdoor").to(dev).eval()

    def prep(img):
        small, s = _resize_cap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), LOFTR_MAX_SIDE)
        # LoFTR needs dimensions divisible by 8.
        h, w = small.shape
        small = small[: h // 8 * 8, : w // 8 * 8]
        t = torch.from_numpy(small).float()[None, None].to(dev) / 255.0
        return t, s

    ta, sa = prep(tgt)
    tb, sb = prep(ref)
    with torch.inference_mode():
        out = _LOFTR({"image0": ta, "image1": tb})
    conf = out["confidence"].cpu().numpy()
    keep = conf >= conf_thr
    pa = out["keypoints0"].cpu().numpy()[keep].astype(np.float64) / sa
    pb = out["keypoints1"].cpu().numpy()[keep].astype(np.float64) / sb
    return pa, pb


# --------------------------------------------------------------------------
# coarse alignment from Face Mesh
# --------------------------------------------------------------------------

_LANDMARKER = None


def landmarks_of(img_path: Path) -> np.ndarray | None:
    global _LANDMARKER
    if _LANDMARKER is None:
        _LANDMARKER = facemesh_check._make_landmarker()
    r = facemesh_check.analyse(img_path, _LANDMARKER)
    return r.lm_px if r.ok else None


def coarse_similarity(tgt_path: Path, ref_path: Path) -> tuple[np.ndarray | None, int]:
    """Similarity taking target -> reference, from Face Mesh landmarks."""
    a, b = landmarks_of(tgt_path), landmarks_of(ref_path)
    if a is None or b is None:
        return None, 0
    M, inl = cv2.estimateAffinePartial2D(
        a.reshape(-1, 1, 2).astype(np.float32),
        b.reshape(-1, 1, 2).astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=8.0, maxIters=5000,
    )
    if M is None:
        return None, 0
    return np.vstack([M, [0, 0, 1]]).astype(np.float64), int(inl.sum()) if inl is not None else 0


# --------------------------------------------------------------------------
# evaluation - the ONLY place control points are touched
# --------------------------------------------------------------------------

@dataclass
class MethodResult:
    name: str
    kind: str
    ok: bool = False
    n_matches: int = 0
    n_inliers: int = 0
    seconds: float = 0.0
    residuals_mm: np.ndarray = field(default_factory=lambda: np.empty(0))
    cp_ids: list[int] = field(default_factory=list)
    ref_pts: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    pred_pts: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    median_mm: float = float("nan")
    p90_mm: float = float("nan")
    p95_mm: float = float("nan")
    max_mm: float = float("nan")
    radial_pearson: float = float("nan")
    radial_p: float = float("nan")
    note: str = ""
    transform: Transform | None = None

    def summary(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "ok": self.ok,
            "n_matches": self.n_matches, "n_inliers": self.n_inliers,
            "seconds": round(self.seconds, 2),
            "median_mm": self.median_mm, "p90_mm": self.p90_mm,
            "p95_mm": self.p95_mm, "max_mm": self.max_mm,
            "radial_pearson": self.radial_pearson, "radial_p": self.radial_p,
            "note": self.note,
        }


def evaluate(
    res: MethodResult,
    T: Transform,
    cps_tgt: dict[int, tuple[float, float]],
    cps_ref: dict[int, tuple[float, float]],
    px_per_mm_ref: float,
    ref_shape: tuple[int, int],
) -> MethodResult:
    """Score a transform against held-out control points. READ ONLY.

    `T` arrives already estimated. Nothing in this function feeds back into
    estimation - that separation is the whole point of the metric.
    """
    ids = sorted(set(cps_tgt) & set(cps_ref))
    if len(ids) < 3:
        res.ok = False
        res.note = f"only {len(ids)} shared control points"
        return res

    src = np.array([cps_tgt[i] for i in ids], np.float64)
    dst = np.array([cps_ref[i] for i in ids], np.float64)
    pred = T.map_points(src)

    d_px = np.linalg.norm(pred - dst, axis=1)
    d_mm = d_px / px_per_mm_ref

    res.ok = True
    res.transform = T
    res.kind = T.kind
    res.cp_ids = ids
    res.ref_pts, res.pred_pts = dst, pred
    res.residuals_mm = d_mm
    res.median_mm = float(np.median(d_mm))
    res.p90_mm = float(np.percentile(d_mm, 90))
    res.p95_mm = float(np.percentile(d_mm, 95))
    res.max_mm = float(d_mm.max())

    # Is the error structured? Homography assumes a plane; a turned face is
    # not one, so the classic signature is error growing toward the edges.
    h, w = ref_shape
    r = np.linalg.norm(dst - np.array([w / 2.0, h / 2.0]), axis=1)
    if len(r) >= 3 and np.ptp(r) > 0:
        from scipy.stats import pearsonr

        pr, pp = pearsonr(r, d_mm)
        res.radial_pearson, res.radial_p = float(pr), float(pp)
    return res


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------

def _homography(pa: np.ndarray, pb: np.ndarray, thr: float = 4.0):
    if len(pa) < 4:
        return None, 0
    H, mask = cv2.findHomography(
        pa.reshape(-1, 1, 2), pb.reshape(-1, 1, 2),
        method=cv2.USAC_MAGSAC, ransacReprojThreshold=thr, maxIters=20000, confidence=0.9999,
    )
    if H is None:
        return None, 0
    return H, int(mask.sum()) if mask is not None else 0


def _spatial_subsample(pa: np.ndarray, pb: np.ndarray, shape, target: int = 500):
    """Grid-thin matches so TPS sees even coverage instead of texture clumps."""
    if len(pa) <= target:
        return pa, pb
    h, w = shape
    g = max(1, int(np.sqrt(target)))
    keys = (pb[:, 1] / h * g).astype(int) * g + (pb[:, 0] / w * g).astype(int)
    keep = []
    for k in np.unique(keys):
        idx = np.where(keys == k)[0]
        keep.extend(idx[: max(1, target // max(1, len(np.unique(keys))))])
    keep = np.array(sorted(keep))[:target]
    return pa[keep], pb[keep]


def run_pair(ref_path: Path, tgt_path: Path, tps_lambda: float = 1e-3) -> dict:
    ref = cv2.imread(str(ref_path))
    tgt = cv2.imread(str(tgt_path))
    if ref is None or tgt is None:
        raise SystemExit(f"could not read {ref_path} or {tgt_path}")
    ref_shape = ref.shape[:2]

    fr = fiducial.detect_fiducial(ref)
    if not fr.found or not fr.px_per_mm:
        raise SystemExit(
            f"no fiducial in the reference image ({fr.note}).\n"
            "Every number in this report is in millimetres and divides by that scale. "
            "Without it there is nothing honest to report."
        )
    px_per_mm = fr.px_per_mm

    cps_ref = load_controlpoints(ref_path)
    cps_tgt = load_controlpoints(tgt_path)

    results: list[MethodResult] = []

    # --- method 0: Face Mesh similarity ----------------------------------
    t0 = time.time()
    Hc, n_inl = coarse_similarity(tgt_path, ref_path)
    r0 = MethodResult("0 facemesh-similarity", "similarity")
    r0.seconds = time.time() - t0
    if Hc is None:
        r0.note = "Face Mesh failed on one of the pair"
        results.append(r0)
    else:
        r0.n_matches = r0.n_inliers = n_inl
        evaluate(r0, HomographyTransform(Hc, "similarity"), cps_tgt, cps_ref, px_per_mm, ref_shape)
        results.append(r0)

    # Coarse-align once, then refine on top of it.
    if Hc is not None:
        tgt_coarse = cv2.warpPerspective(tgt, Hc, (ref.shape[1], ref.shape[0]))
    else:
        tgt_coarse = tgt

    def refined(name, matcher, thr=4.0):
        t = time.time()
        r = MethodResult(name, "homography")
        try:
            pa, pb = matcher(tgt_coarse, ref)
        except Exception as exc:  # noqa: BLE001
            r.note = f"matcher failed: {exc!r}"
            r.seconds = time.time() - t
            return r, None, None
        r.n_matches = len(pa)
        Hf, inl = _homography(pa, pb, thr)
        r.n_inliers = inl
        r.seconds = time.time() - t
        if Hf is None:
            r.note = f"homography failed from {len(pa)} matches"
            return r, None, None
        total = Hf @ (Hc if Hc is not None else np.eye(3))
        evaluate(r, HomographyTransform(total), cps_tgt, cps_ref, px_per_mm, ref_shape)
        return r, (pa, pb), total

    r1, m1, _ = refined("1 coarse+SIFT+RANSAC", match_sift)
    results.append(r1)
    r2, m2, _ = refined("2 coarse+ORB+RANSAC", match_orb)
    results.append(r2)
    r3, m3, _ = refined("3 coarse+LightGlue+DISK", match_disk_lightglue)
    results.append(r3)

    # --- method 4: LoFTR, no coarse step ---------------------------------
    t = time.time()
    r4 = MethodResult("4 LoFTR (detector-free)", "homography")
    m4 = None
    try:
        pa, pb = match_loftr(tgt, ref)
        r4.n_matches = len(pa)
        H4, inl4 = _homography(pa, pb, 4.0)
        r4.n_inliers = inl4
        if H4 is not None:
            evaluate(r4, HomographyTransform(H4), cps_tgt, cps_ref, px_per_mm, ref_shape)
            m4 = (pa, pb)
        else:
            r4.note = f"homography failed from {len(pa)} matches"
    except Exception as exc:  # noqa: BLE001
        r4.note = f"LoFTR failed: {exc!r}"
    r4.seconds = time.time() - t
    results.append(r4)

    # --- method 5: TPS on the best method's matches ----------------------
    # "Best" is chosen by median residual. That uses control points to SELECT
    # a method, not to FIT one - the TPS itself sees only feature matches.
    cand = [
        (r1, m1, Hc), (r2, m2, Hc), (r3, m3, Hc),
        (r4, m4, np.eye(3)),
    ]
    scored = [(r.median_mm, r, m, pre) for r, m, pre in cand if r.ok and m is not None]
    r5 = MethodResult("5 best + thin-plate spline", "tps")
    if not scored:
        r5.note = "no successful matcher to build a TPS from"
    else:
        scored.sort(key=lambda s: s[0])
        _, base, (pa, pb), pre = scored[0]
        t = time.time()
        try:
            # Matches live in the coarse-warped frame; push them back to raw
            # target coordinates so the TPS maps target -> reference directly.
            pre_inv = np.linalg.inv(pre)
            pa_raw = cv2.perspectiveTransform(pa.reshape(-1, 1, 2), pre_inv).reshape(-1, 2)

            H_rough, mask = cv2.findHomography(
                pa_raw.reshape(-1, 1, 2), pb.reshape(-1, 1, 2),
                method=cv2.USAC_MAGSAC, ransacReprojThreshold=6.0, maxIters=20000,
            )
            if mask is not None:
                keep = mask.ravel().astype(bool)
                pa_raw, pb = pa_raw[keep], pb[keep]

            sa, sb = _spatial_subsample(pa_raw, pb, ref_shape, target=500)
            diag = float(np.hypot(*ref_shape))
            fwd = TPSTransform(sa, sb, lam=tps_lambda, scale=diag)
            inv = TPSTransform(sb, sa, lam=tps_lambda, scale=diag)
            r5.name = f"5 TPS on {base.name.split(' ', 1)[1]}"
            r5.n_matches, r5.n_inliers = len(pa_raw), len(sa)
            evaluate(r5, TPSPair(fwd, inv), cps_tgt, cps_ref, px_per_mm, ref_shape)
        except Exception as exc:  # noqa: BLE001
            r5.note = f"TPS failed: {exc!r}"
        r5.seconds = time.time() - t
    results.append(r5)

    return {
        "reference": ref_path,
        "target": tgt_path,
        "px_per_mm": px_per_mm,
        "ref_shape": ref_shape,
        "n_control_points": len(set(cps_ref) & set(cps_tgt)),
        "results": results,
    }


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def plot_residuals(pair: dict, out_dir: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in pair["results"] if r.ok]
    if not ok:
        return None
    ref = cv2.imread(str(pair["reference"]))
    rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

    n = len(ok)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6.6 * rows), squeeze=False)

    # One exaggeration factor across every panel, or the panels lie by comparison.
    worst = max(r.max_mm for r in ok)
    amp = (0.12 * min(pair["ref_shape"])) / max(worst * pair["px_per_mm"], 1e-6)

    for ax, r in zip(axes.ravel(), ok):
        ax.imshow(rgb)
        d = r.pred_pts - r.ref_pts
        ax.quiver(
            r.ref_pts[:, 0], r.ref_pts[:, 1], d[:, 0] * amp, d[:, 1] * amp,
            color="red", angles="xy", scale_units="xy", scale=1, width=0.005,
        )
        ax.scatter(r.ref_pts[:, 0], r.ref_pts[:, 1], s=28, facecolors="none",
                   edgecolors="cyan", linewidths=1.4)
        ax.set_title(
            f"{r.name}\nmedian {r.median_mm:.2f} mm   p95 {r.p95_mm:.2f} mm   "
            f"inliers {r.n_inliers}\nradial r={r.radial_pearson:+.2f}",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Residual vectors  (arrows x{amp:.0f})   "
        f"{Path(pair['target']).name} -> {Path(pair['reference']).name}   "
        f"{pair['px_per_mm']:.2f} px/mm",
        fontsize=11,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"residuals_{Path(pair['target']).stem}.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def write_report(pairs: list[dict], out: Path, plots: list[Path]) -> None:
    L: list[str] = ["# Registration bake-off - phase 0 question (b)\n"]

    names: list[str] = []
    for p in pairs:
        for r in p["results"]:
            if r.name not in names:
                names.append(r.name)

    agg: dict[str, list[MethodResult]] = {n: [] for n in names}
    for p in pairs:
        for r in p["results"]:
            if r.ok:
                agg[r.name].append(r)

    L.append("## VERDICT\n")
    scored = [
        (float(np.median([r.median_mm for r in v])), float(np.median([r.p95_mm for r in v])), k)
        for k, v in agg.items() if v
    ]
    if not scored:
        L.append("**NO-GO - nothing scored.** No method produced a transform that could be "
                 "evaluated. Check that control points exist for both images and that the "
                 "fiducial is detected in the reference.\n")
    else:
        scored.sort()
        med, p95, best = scored[0]
        passed = med < config.GATE_MEDIAN_MM and p95 < config.GATE_P95_MM
        L.append(f"- **Gate:** median < {config.GATE_MEDIAN_MM} mm and p95 < {config.GATE_P95_MM} mm")
        L.append(f"- **Best method:** `{best}` - median {med:.2f} mm, p95 {p95:.2f} mm")
        L.append(f"\n**{'GO' if passed else 'NO-GO'}.**")
        if passed:
            L.append(f"`{best}` clears the gate. Use it for the phase 3 pipeline.\n")
        else:
            fails = []
            if med >= config.GATE_MEDIAN_MM:
                fails.append(f"median {med:.2f} mm >= {config.GATE_MEDIAN_MM} mm")
            if p95 >= config.GATE_P95_MM:
                fails.append(f"p95 {p95:.2f} mm >= {config.GATE_P95_MM} mm")
            L.append("Failed on " + " and ".join(fails) + ".")
            L.append("\nCheapest next lever, in order:\n")
            L.append("1. Check the blink compare before believing any number here. If moles visibly "
                     "jitter, the warp is wrong in a way the summary statistics are hiding.")
            L.append("2. If residuals are radially structured (see the correlation column), the "
                     "planar assumption is the problem - TPS is the fix, not a better matcher.")
            L.append("3. If inlier counts are low across every method, the problem is the photos: "
                     "depth of field, motion blur, or lighting change between sessions.")
            L.append("4. Only after those, reduce yaw to 50 deg and re-shoot.\n")

    L.append("\n## Method comparison\n")
    L.append("Medians across all reference/target pairs. Errors in millimetres at held-out "
             "control points, converted through the reference image's own ArUco scale.\n")
    L.append("| method | kind | median mm | p90 mm | p95 mm | max mm | matches | inliers | radial r | sec |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for n in names:
        v = agg[n]
        if not v:
            bad = [r for p in pairs for r in p["results"] if r.name == n]
            note = bad[0].note if bad else "no result"
            L.append(f"| {n} | - | - | - | - | - | - | - | - | - |  <!-- {note} -->")
            continue
        f = lambda a: float(np.median([getattr(r, a) for r in v]))  # noqa: E731
        L.append(
            f"| {n} | {v[0].kind} | {f('median_mm'):.2f} | {f('p90_mm'):.2f} | "
            f"{f('p95_mm'):.2f} | {f('max_mm'):.2f} | {int(f('n_matches'))} | "
            f"{int(f('n_inliers'))} | {f('radial_pearson'):+.2f} | {f('seconds'):.1f} |"
        )

    L.append("\n`radial r` is the Pearson correlation between residual magnitude and distance "
             "from the image centre. Strongly positive means the error grows toward the edges - "
             "the signature of fitting a plane to a face that is not one, and the evidence that "
             "a thin-plate spline is needed rather than a better feature matcher.\n")

    L.append("\n## Per pair\n")
    for p in pairs:
        L.append(f"\n### {Path(p['target']).name} -> {Path(p['reference']).name}\n")
        L.append(f"- scale: **{p['px_per_mm']:.2f} px/mm** (reference image ArUco)")
        L.append(f"- control points: **{p['n_control_points']}** shared, held out")
        L.append("")
        L.append("| method | median mm | p90 mm | p95 mm | max mm | matches | inliers | radial r | p | note |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in p["results"]:
            if not r.ok:
                L.append(f"| {r.name} | - | - | - | - | {r.n_matches} | {r.n_inliers} | - | - | {r.note} |")
                continue
            L.append(
                f"| {r.name} | {r.median_mm:.2f} | {r.p90_mm:.2f} | {r.p95_mm:.2f} | "
                f"{r.max_mm:.2f} | {r.n_matches} | {r.n_inliers} | "
                f"{r.radial_pearson:+.2f} | {r.radial_p:.3f} | {r.note or 'ok'} |"
            )

    if plots:
        L.append("\n## Residual vector plots\n")
        for p in plots:
            L.append(f"- `{p.relative_to(config.REPO) if config.REPO in p.parents else p}`")

    L.append("\n## Read this before trusting the table\n")
    L.append("- Control points were used for scoring only. No transform in this report was fitted "
             "using them; method 5 uses them to *pick* which matcher to spline, never to fit.")
    L.append("- Summary statistics can look fine while the warp is subtly wrong. Run the blink "
             "compare on the winning method before declaring the gate passed:\n")
    L.append("  ```\n  python -m src.register review --ref REF.jpg --target T.jpg --method tps\n  ```")
    L.append("- With ~10 control points, p95 is one or two points. Treat it as directional.\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# review / blink compare
# --------------------------------------------------------------------------

def review(ref_path: Path, tgt_path: Path, method: str = "tps", hz: float = 2.0) -> int:
    """Side-by-side plus blink compare. The honest check on a warp."""
    pair = run_pair(ref_path, tgt_path)
    ok = [r for r in pair["results"] if r.ok]
    if not ok:
        print("no method produced an evaluable transform")
        return 1

    want = [r for r in ok if method.lower() in r.name.lower() or method.lower() == r.kind]
    r = (want or sorted(ok, key=lambda x: x.median_mm))[0]
    print(f"reviewing: {r.name}")

    ref = cv2.imread(str(ref_path))
    tgt = cv2.imread(str(tgt_path))
    warped = r.transform.warp_to_reference(tgt, pair["ref_shape"])

    fm_ok = "OK" if any(m.name.startswith("0") and m.ok for m in pair["results"]) else "FAILED"
    readout = (
        f"median {r.median_mm:.2f} mm | p95 {r.p95_mm:.2f} mm | matches {r.n_inliers} | "
        f"{r.kind} | {pair['px_per_mm']:.2f} px/mm | FaceMesh {fm_ok}"
    )

    def annotate(img, title):
        v = img.copy()
        for (rx, ry), (px_, py) in zip(r.ref_pts, r.pred_pts):
            cv2.circle(v, (int(rx), int(ry)), 9, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.drawMarker(v, (int(px_), int(py)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(v, title, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(v, title, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        return v

    a, b = annotate(ref, "REFERENCE (atlas)"), annotate(warped, "TARGET warped to reference")

    scr_w = 1500
    sbs = np.hstack([a, b])
    s = scr_w / sbs.shape[1]
    sbs_small = cv2.resize(sbs, None, fx=s, fy=s)
    bar = np.zeros((46, sbs_small.shape[1], 3), np.uint8)
    cv2.putText(bar, readout, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    sbs_small = np.vstack([sbs_small, bar])

    sb = scr_w // 2
    sa_ = sb / ref.shape[1]
    blink = [cv2.resize(a, None, fx=sa_, fy=sa_), cv2.resize(b, None, fx=sa_, fy=sa_)]
    bbar = np.zeros((46, blink[0].shape[1], 3), np.uint8)
    cv2.putText(bbar, readout, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    blink = [np.vstack([x, bbar]) for x in blink]

    cv2.namedWindow("side by side", cv2.WINDOW_NORMAL)
    cv2.namedWindow("BLINK COMPARE", cv2.WINDOW_NORMAL)
    cv2.imshow("side by side", sbs_small)
    print("\n  BLINK COMPARE running at "
          f"{hz:.1f} Hz. Watch the moles - if they jitter, the warp is wrong")
    print("  regardless of what the millimetres say.")
    print("  [space] pause/resume   [n] step   [q/esc] quit\n")

    i, paused, period = 0, False, 1.0 / (2.0 * hz)
    last = time.time()
    while True:
        cv2.imshow("BLINK COMPARE", blink[i % 2])
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord(" "):
            paused = not paused
        if k == ord("n"):
            i += 1
        if not paused and time.time() - last >= period:
            i += 1
            last = time.time()
        if cv2.getWindowProperty("BLINK COMPARE", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    return 0


# --------------------------------------------------------------------------

def _collect(targets: list[Path]) -> list[Path]:
    out = []
    for t in targets:
        if t.is_dir():
            out.extend(sorted(p for p in t.iterdir() if p.suffix.lower() in IMG_EXTS))
        elif t.suffix.lower() in IMG_EXTS:
            out.append(t)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="registration bake-off")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bakeoff")
    b.add_argument("--ref", type=Path, required=True)
    b.add_argument("--targets", type=Path, nargs="+", required=True)
    b.add_argument("--report", type=Path, default=config.REPORTS_DIR / "registration_report.md")
    b.add_argument("--plots", type=Path, default=config.REPORTS_DIR / "registration_plots")
    b.add_argument("--tps-lambda", type=float, default=1e-3)
    b.add_argument("--cp-dir", type=Path, default=None,
                   help="control-point directory (default: the live data root)")

    r = sub.add_parser("review")
    r.add_argument("--ref", type=Path, required=True)
    r.add_argument("--target", type=Path, required=True)
    r.add_argument("--method", default="tps")
    r.add_argument("--hz", type=float, default=2.0)
    r.add_argument("--cp-dir", type=Path, default=None)

    a = ap.parse_args(argv)
    if getattr(a, "cp_dir", None):
        set_controlpoints_dir(a.cp_dir)
    if a.cmd == "review":
        return review(a.ref, a.target, a.method, a.hz)

    tgts = [p for p in _collect(a.targets) if p.resolve() != a.ref.resolve()]
    if not tgts:
        print("no target images found")
        return 2

    pairs, plots = [], []
    for t in tgts:
        print(f"\n=== {t.name} -> {a.ref.name} ===")
        p = run_pair(a.ref, t, a.tps_lambda)
        for res in p["results"]:
            if res.ok:
                print(f"  {res.name:32s} median {res.median_mm:6.2f} mm   "
                      f"p95 {res.p95_mm:6.2f} mm   inliers {res.n_inliers:5d}   {res.seconds:5.1f}s")
            else:
                print(f"  {res.name:32s} FAILED  {res.note}")
        pairs.append(p)
        pl = plot_residuals(p, a.plots)
        if pl:
            plots.append(pl)

    write_report(pairs, a.report, plots)
    j = a.report.with_suffix(".json")
    j.write_text(
        json.dumps(
            [
                {
                    "reference": str(p["reference"]), "target": str(p["target"]),
                    "px_per_mm": p["px_per_mm"], "n_control_points": p["n_control_points"],
                    "results": [r.summary() for r in p["results"]],
                }
                for p in pairs
            ],
            indent=2, default=float,
        ),
        encoding="utf-8",
    )
    print(f"wrote {j}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
