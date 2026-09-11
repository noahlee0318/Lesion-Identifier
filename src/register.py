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
            run once per pretrained prior (indoor / outdoor); both are
            reported, neither is assumed
  method 5  best of the above, then thin-plate spline on its matched points

GATE: median < 1.5 mm and p95 < 3 mm, on residuals POOLED ACROSS ALL PAIRS.

=============================================================================
HOW THE GATE NUMBER IS COMPUTED, AND WHY IT IS NOT THE MINIMUM

Two things would make the headline figure optimistically biased if they were
done the obvious way, and both are avoided deliberately:

1. POOLING, NOT MEDIAN-OF-MEDIANS. Every residual from every pair goes into
   one vector per method, and the gate statistics are computed on that vector.
   A median of per-pair medians is a different and weaker statistic: it is
   insensitive to one catastrophic pair, which is precisely the failure the
   gate exists to catch. Per-pair numbers are still reported as a breakdown.

   A method is only eligible for the gate if it scored on EVERY pair. A method
   that fails outright on the hardest pair contributes no residuals from it,
   so its pool is drawn from the easy pairs and it would otherwise be rewarded
   for failing.

2. RECOMMEND THE SIMPLEST METHOD THAT CLEARS, NOT THE LOWEST NUMBER. With six
   or seven candidates scored on ~50 pooled control points, the minimum is
   meaningfully below the true out-of-sample error. A method that wins by
   0.05 mm has not earned its extra machinery. Ties inside a complexity tier
   are broken by pooled median, which is the only place "lowest" is used.

The GO/NO-GO decision is still a max over candidates - "did anything clear" -
so the report states out loud that the winning figure is selected-best-of-N on
a small sample and the honest read on new data is somewhat worse.
=============================================================================

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
from typing import Sequence

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

# LoFTR ships two priors. "outdoor" was the original constant here; close-range
# faces are arguably closer to the indoor training distribution, so it is a
# parameter and both are run in the bake-off rather than one being assumed.
LOFTR_WEIGHTS = ("indoor", "outdoor")

# A percentile is only meaningful if at least one sample sits above it:
# n * (1 - q/100) >= 1. That puts p95 at n >= 20 and p90 at n >= 10. Below
# that, np.percentile is interpolating between the top two values and is just
# the maximum wearing a percentile's name - so it is refused, not printed.
def min_n_for_percentile(q: float) -> int:
    """Smallest sample size at which percentile `q` is not just the maximum."""
    # The 1e-9 guard is float hygiene, not slack: 1/(1-90/100) evaluates to
    # 10.000000000000002 in binary floating point, and a bare ceil would
    # demand 11 samples for a p90.
    return int(np.ceil(1.0 / (1.0 - q / 100.0) - 1e-9))


# --------------------------------------------------------------------------
# method identity and the complexity ladder
# --------------------------------------------------------------------------
#
# A method's NAME is its pooling key, so it must be identical across pairs.
# (Method 5 used to rename itself after whichever matcher it splined, which
# would have split one method into several one-pair rows the moment two pairs
# picked different bases. The base now goes in the note instead.)

M0 = "0 facemesh-similarity"
M1 = "1 coarse+SIFT+RANSAC"
M2 = "2 coarse+ORB+RANSAC"
M3 = "3 coarse+LightGlue+DISK"
M5 = "5 best + thin-plate spline"


def loftr_name(pretrained: str) -> str:
    return f"4 LoFTR-{pretrained} (detector-free)"


# Complexity tiers. The recommendation is the SIMPLEST method that clears the
# gate, not the lowest number, so this ordering is load-bearing: it decides
# what gets recommended whenever more than one method passes.
#
#   0  similarity from Face Mesh landmarks        - 4 dof, no matcher at all
#   1  homography from a classical detector       - SIFT / ORB, CPU, no weights
#   2  homography from a learned detector+matcher - DISK + LightGlue, GPU
#   3  homography from a detector-free matcher    - LoFTR, GPU, 6 GiB-bound
#   4  thin-plate spline                          - non-rigid, 500-point solve
#
# Ties INSIDE a tier are broken by pooled median. That is the only place a
# "lowest number" comparison is allowed to pick anything.
_TIER_OF_DIGIT = {"0": 0, "1": 1, "2": 1, "3": 2, "4": 3, "5": 4}
TIER_LABEL = {
    0: "similarity",
    1: "homography, classical detector",
    2: "homography, learned detector",
    3: "homography, detector-free",
    4: "thin-plate spline",
}

# A TPS fitted on a handful of points is not a spline, it is noise with extra
# steps. Below this a matcher is simply not a candidate base for method 5.
MIN_TPS_INLIERS = 25


def method_tier(name: str) -> int:
    """Complexity tier of a method, from its leading number."""
    return _TIER_OF_DIGIT.get(name.split(" ", 1)[0], 99)


class RegistrationInputError(RuntimeError):
    """A pair cannot be scored because of its inputs, not because of a bug.

    Missing image, missing fiducial, unreadable file. Raised rather than
    exiting so that a batch run can skip the pair and the function stays
    testable; the CLI turns it into an exit code.
    """


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

    # Filled in by warp_to_reference for transforms whose dense inverse is
    # approximate. None means "the inverse used for the picture is the exact
    # algebraic inverse of the map that produced the numbers".
    last_roundtrip: dict | None = None

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


def _tps_kernel_grad_factor(r2: np.ndarray) -> np.ndarray:
    """g(r2) with  d/dq U(|q-c|^2) = g(r2) * (q - c).

    U(r2) = r2*log(r2)  =>  dU/dr2 = log(r2) + 1, and d(r2)/dq = 2(q - c),
    so g = 2*(log(r2) + 1). At r2 = 0 the true gradient is 0; the factor is
    forced to 0 there so that 0 * -inf does not become NaN.
    """
    out = np.zeros_like(r2)
    nz = r2 > 1e-12
    out[nz] = 2.0 * (np.log(r2[nz]) + 1.0)
    return out


# Grid points processed per batch when mapping or inverting. The dense kernel
# is (points x control points x 2) float64; a 240-step grid over a 4032x3024
# frame is ~77k points, which at 500 control points would be 600 MB in one go.
_TPS_CHUNK = 8192


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
        q = np.asarray(pts, np.float64).reshape(-1, 2)
        out = np.empty_like(q)
        for lo in range(0, len(q), _TPS_CHUNK):
            out[lo:lo + _TPS_CHUNK] = self._map_chunk(q[lo:lo + _TPS_CHUNK] / self.scale)
        return out * self.scale

    def _map_chunk(self, qn: np.ndarray) -> np.ndarray:
        """Map already-normalised points. Returns normalised output."""
        diff = qn[:, None, :] - self.src[None, :, :]
        U = _tps_kernel((diff ** 2).sum(-1))
        P = np.hstack([np.ones((len(qn), 1)), qn])
        n = len(self.src)
        return U @ self.W[:n] + P @ self.W[n:]

    def _map_and_jac_chunk(self, qn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalised map plus its 2x2 Jacobian at each point.

        The Jacobian is identical in normalised and pixel coordinates: input
        and output are divided by the same scale, so the factors cancel.
        """
        n = len(self.src)
        diff = qn[:, None, :] - self.src[None, :, :]        # (m, n, 2)
        r2 = (diff ** 2).sum(-1)                            # (m, n)
        U = _tps_kernel(r2)
        g = _tps_kernel_grad_factor(r2)
        P = np.hstack([np.ones((len(qn), 1)), qn])
        Wn, Wa = self.W[:n], self.W[n:]

        out = U @ Wn + P @ Wa
        # J[m, k, j] = sum_i Wn[i, k] * g[m, i] * diff[m, i, j] + Wa[1 + j, k]
        gd = g[:, :, None] * diff                           # (m, n, 2)
        J = np.einsum("mij,ik->mkj", gd, Wn)                # (m, 2, 2)
        J += Wa[1:].T[None, :, :]                           # affine part
        return out, J

    def invert_points(self, pts: np.ndarray, seed: np.ndarray, iters: int = 3) -> np.ndarray:
        """Newton solve of  fwd(q) = p,  started from `seed`. Pixel units.

        `seed` comes from the independently fitted reverse spline, which is a
        good initialisation and a bad inverse (see TPSPair). Two or three
        Newton steps turn it into an actual inverse.

        Points where the Jacobian is near-singular keep the seed value: TPS
        folds over on itself only well outside the matched-point hull, where
        the warp is extrapolation and meaningless either way.
        """
        target = np.asarray(pts, np.float64).reshape(-1, 2) / self.scale
        q = np.asarray(seed, np.float64).reshape(-1, 2).copy() / self.scale

        for _ in range(max(0, iters)):
            for lo in range(0, len(q), _TPS_CHUNK):
                sl = slice(lo, lo + _TPS_CHUNK)
                f, J = self._map_and_jac_chunk(q[sl])
                res = f - target[sl]
                det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
                ok = np.abs(det) > 1e-12
                step = np.zeros_like(res)
                d = det[ok]
                a, b, c, e = J[ok, 0, 0], J[ok, 0, 1], J[ok, 1, 0], J[ok, 1, 1]
                rx, ry = res[ok, 0], res[ok, 1]
                step[ok, 0] = (e * rx - b * ry) / d
                step[ok, 1] = (-c * rx + a * ry) / d
                q[sl] -= step
        return q * self.scale

    def warp_to_reference(self, img, out_hw):
        # Dense remap needs the INVERSE map (ref pixel -> target pixel), so the
        # caller supplies an inverse-fitted twin; see TPSPair.
        raise NotImplementedError("use TPSPair.warp_to_reference")


class TPSPair(Transform):
    """Forward TPS for points, Newton-refined inverse for image warping.

    WHICH INVERSE THIS USES, AND WHY IT MATTERS

    The numbers in the report come from `fwd`. The picture in the blink
    compare comes from the dense inverse. If those two are different maps, the
    reviewer is checking one transform by eye and reading the millimetres of
    another - and the blink compare is the honest test precisely because
    summary statistics can look fine while the warp is wrong.

    A separately fitted reverse spline TPSTransform(dst, src) is NOT the
    inverse of TPSTransform(src, dst). Both are regularised (lam = 1e-3 by
    default), both are deliberately smoothed, and the composition drifts.

    So option (a) of the review is taken: the reverse fit is used only as a
    SEED, and each grid point is then Newton-refined against `fwd` until it
    really is fwd's preimage. The residual  |fwd(inv(p)) - p|  is measured
    over the grid and stored in `last_roundtrip` (pixels), so the review
    screen can state how far the picture is from the numbers instead of
    assuming it is zero.
    """

    kind = "tps"

    def __init__(self, fwd: TPSTransform, inv: TPSTransform, newton_iters: int = 3):
        self.fwd, self.inv = fwd, inv
        self.newton_iters = int(newton_iters)
        self.last_roundtrip: dict | None = None

    def map_points(self, pts):
        return self.fwd.map_points(pts)

    def inverse_map(self, pts: np.ndarray, with_seed: bool = False):
        """REFERENCE points -> TARGET points. The actual inverse of map_points.

        With `with_seed`, also returns the unrefined reverse-fit result, so a
        caller can report what the refinement was worth.
        """
        seed = self.inv.map_points(pts)
        if self.newton_iters <= 0:
            out = seed
        else:
            out = self.fwd.invert_points(pts, seed, iters=self.newton_iters)
        return (out, seed) if with_seed else out

    def _hull_mask(self, pts: np.ndarray) -> np.ndarray:
        """Which of `pts` lie inside the hull of the reference-side TPS points.

        Outside that hull the spline is extrapolating and both the warp and
        its inverse are meaningless, so round-trip error there says nothing
        about the region anyone looks at.
        """
        ref_pts = (self.inv.src * self.inv.scale).astype(np.float32)
        if len(ref_pts) < 3:
            return np.ones(len(pts), bool)
        hull = cv2.convexHull(ref_pts.reshape(-1, 1, 2))
        return np.array(
            [cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0 for x, y in pts],
            dtype=bool,
        )

    def warp_to_reference(self, img, out_hw):
        h, w = out_hw
        step = max(1, int(min(h, w) / 240))       # solve on a coarse grid, upsample
        ys = np.arange(0, h, step, dtype=np.float64)
        xs = np.arange(0, w, step, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)
        grid = np.column_stack([gx.ravel(), gy.ravel()])

        mapped, seed = self.inverse_map(grid, with_seed=True)

        # Measure what the picture actually is: push the inverse back through
        # the forward map and see where it lands relative to where it started.
        # `seed_max_px` is the same measurement on the UNREFINED reverse fit -
        # what the blink compare would have been showing before, and the number
        # that says whether the refinement was worth anything on this pair.
        err = np.linalg.norm(self.fwd.map_points(mapped) - grid, axis=1)
        seed_err = np.linalg.norm(self.fwd.map_points(seed) - grid, axis=1)
        inside = self._hull_mask(grid)
        self.last_roundtrip = {
            "n_grid": int(len(grid)),
            "n_inside_hull": int(inside.sum()),
            "newton_iters": self.newton_iters,
            "max_px": float(err.max()) if len(err) else float("nan"),
            "p95_px": float(np.percentile(err, 95)) if len(err) else float("nan"),
            "median_px": float(np.median(err)) if len(err) else float("nan"),
            "max_in_hull_px": float(err[inside].max()) if inside.any() else float("nan"),
            "seed_max_px": float(seed_err.max()) if len(seed_err) else float("nan"),
        }

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


_DISK = _LG = None
_LOFTR: dict[str, object] = {}


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


def match_loftr(tgt: np.ndarray, ref: np.ndarray, conf_thr: float = 0.5,
                pretrained: str = "outdoor"):
    """LoFTR matches. `pretrained` selects the prior - see LOFTR_WEIGHTS.

    Which prior suits a close-range face is an empirical question, not a
    constant: the bake-off runs both and the report compares them on the
    pooled metric like any other candidate.
    """
    import torch
    from kornia.feature import LoFTR

    config.torch_home()
    dev = _device()
    if pretrained not in _LOFTR:
        _LOFTR[pretrained] = LoFTR(pretrained=pretrained).to(dev).eval()
    model = _LOFTR[pretrained]

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
        out = model({"image0": ta, "image1": tb})
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
    # Properties of the MATCH SET, not of the held-out truth. Used to pick
    # method 5's base matcher without spending any of the test set.
    inlier_ratio: float = float("nan")
    cell_coverage: float = float("nan")

    @property
    def n_residuals(self) -> int:
        return int(len(self.residuals_mm))

    def summary(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "ok": self.ok,
            "n_matches": self.n_matches, "n_inliers": self.n_inliers,
            "n_residuals": self.n_residuals,
            "seconds": round(self.seconds, 2),
            "median_mm": self.median_mm, "p90_mm": self.p90_mm,
            "p95_mm": self.p95_mm, "max_mm": self.max_mm,
            "inlier_ratio": self.inlier_ratio, "cell_coverage": self.cell_coverage,
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
    # Per-pair percentiles on ~10 points are the top one or two residuals
    # wearing a percentile's name. They are computed for the per-pair
    # breakdown but are NOT what the gate reads - see pool_by_method.
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


def _cell_keys(pb: np.ndarray, shape, g: int) -> np.ndarray:
    """Which grid cell each reference-side point falls in."""
    h, w = shape
    cy = np.clip((pb[:, 1] / h * g).astype(int), 0, g - 1)
    cx = np.clip((pb[:, 0] / w * g).astype(int), 0, g - 1)
    return cy * g + cx


def _spatial_subsample(pa: np.ndarray, pb: np.ndarray, shape, target: int = 500):
    """Grid-thin matches so TPS sees even coverage instead of texture clumps.

    Selection is ROUND-ROBIN across occupied cells: one point from each cell,
    then a second from every cell that still has one, and so on until `target`
    is reached. Any truncation is therefore itself spatially even, at whatever
    grid resolution and target a future caller picks.

    This replaces a per-cell quota followed by `sorted(keep)[:target]`. Two
    things were wrong with that, only one of them reachable:

      * REACHABLE: the quota was `target // n_occupied_cells`, and the grid is
        `int(sqrt(target))` on a side, so there are at most `target` cells and
        the quota collapses to 1. It returned ONE point per occupied cell -
        at most 484 of the 500 asked for - and capped the spline's effective
        resolution at the thinning grid no matter how many good matches the
        matcher found.

      * LATENT: the final `sorted(keep)[:target]` truncates by ARRAY INDEX,
        which is detector output order, not position - it would have undone
        the whole point of the function. It never fires at the current
        parameters (n_cells * quota <= target always), but it fires the moment
        anyone makes the grid finer than sqrt(target), and it would fail
        silently and look like a matcher problem.
    """
    if len(pa) <= target:
        return pa, pb

    g = max(1, int(np.sqrt(target)))
    keys = _cell_keys(pb, shape, g)

    by_cell: dict[int, list[int]] = {}
    for i, k in enumerate(keys):
        by_cell.setdefault(int(k), []).append(i)

    cells = sorted(by_cell)                      # deterministic cell order
    keep: list[int] = []
    depth = 0
    while len(keep) < target:
        added = False
        for c in cells:
            if depth < len(by_cell[c]):
                keep.append(by_cell[c][depth])
                added = True
                if len(keep) == target:
                    break
        if not added:                            # every cell exhausted
            break
        depth += 1

    idx = np.array(sorted(keep))
    return pa[idx], pb[idx]


def _match_set_quality(pa: np.ndarray, pb: np.ndarray, n_inliers: int, shape) -> tuple[float, float]:
    """(inlier_ratio, cell_coverage) for a match set. No control points.

    `cell_coverage` is the fraction of an 8x8 grid over the reference frame
    that holds at least one match - a direct measure of whether a spline
    fitted on these points will be constrained everywhere or only where the
    texture happens to be.
    """
    if len(pa) == 0:
        return float("nan"), float("nan")
    ratio = float(n_inliers) / float(len(pa))
    g = 8
    cells = np.unique(_cell_keys(pb, shape, g))
    return ratio, float(len(cells)) / float(g * g)


def _pick_tps_base(cand: list) -> tuple:
    """Choose which matcher method 5 splines. MATCH-SET PROPERTIES ONLY.

    The previous rule picked the base by control-point median and then scored
    the result on those same control points: selection on the test set, on ~10
    points, which biases method 5's number downward for free.

    Coverage, inlier ratio and inlier count are all properties of the match
    set. They are exactly what determines whether a thin-plate spline is well
    posed, they are available without looking at the held-out truth, and using
    them therefore costs nothing.

    Coverage leads because it is what TPS is sensitive to. Raw inlier COUNT is
    deliberately last and only a tiebreak: it rewards dense matchers (LoFTR
    routinely returns 8000 matches to SIFT's 3800) for being dense rather than
    for being right.
    """
    usable = [c for c in cand if c[1] is not None and c[0].n_inliers >= MIN_TPS_INLIERS]
    if not usable:
        return ()

    def key(c):
        r = c[0]
        cov = 0.0 if np.isnan(r.cell_coverage) else r.cell_coverage
        rat = 0.0 if np.isnan(r.inlier_ratio) else r.inlier_ratio
        # Coarse coverage bands, so a one-cell difference cannot outrank a
        # visibly better inlier ratio.
        return (-round(cov, 1), -round(rat, 2), -r.n_inliers, r.name)

    usable.sort(key=key)
    return usable[0]


def run_pair(
    ref_path: Path,
    tgt_path: Path,
    tps_lambda: float = 1e-3,
    loftr_weights: Sequence[str] = LOFTR_WEIGHTS,
) -> dict:
    """Run the full ladder on one reference/target pair.

    Raises RegistrationInputError if the pair cannot be scored at all - an
    unreadable image, or a reference with no fiducial. Method-level failures
    are recorded on the MethodResult instead, so one broken matcher never
    takes down the run.
    """
    ref = cv2.imread(str(ref_path))
    tgt = cv2.imread(str(tgt_path))
    if ref is None or tgt is None:
        missing = ref_path if ref is None else tgt_path
        raise RegistrationInputError(f"could not read image: {missing}")
    ref_shape = ref.shape[:2]

    fr = fiducial.detect_fiducial(ref)
    if not fr.found or not fr.px_per_mm:
        raise RegistrationInputError(
            f"no fiducial in the reference image {ref_path.name} ({fr.note}).\n"
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
    r0 = MethodResult(M0, "similarity")
    r0.seconds = time.time() - t0
    if Hc is None:
        r0.note = "Face Mesh failed on one of the pair"
        results.append(r0)
    else:
        r0.n_matches = r0.n_inliers = n_inl
        evaluate(r0, HomographyTransform(Hc, "similarity"), cps_tgt, cps_ref, px_per_mm, ref_shape)
        results.append(r0)

    # Coarse-align once, then refine on top of it.
    coarse_ok = Hc is not None
    tgt_coarse = cv2.warpPerspective(tgt, Hc, (ref.shape[1], ref.shape[0])) if coarse_ok else tgt

    def refined(name, matcher, thr=4.0):
        """Match against the coarse-aligned target, then fit a homography.

        If Face Mesh failed there IS no coarse step, and what actually ran is
        plain SIFT/ORB/LightGlue on unaligned frames - a different method with
        a harder job. That is recorded in the note and reflected in the
        reported name, rather than being reported under a name claiming a
        stage that did not happen.
        """
        t = time.time()
        r = MethodResult(name, "homography")
        if not coarse_ok:
            r.name = name.replace("coarse+", "no-coarse+")
            r.note = "Face Mesh failed: no coarse step, matched on unaligned frames"
        try:
            pa, pb = matcher(tgt_coarse, ref)
        except Exception as exc:  # noqa: BLE001
            r.note = (r.note + "; " if r.note else "") + f"matcher failed: {exc!r}"
            r.seconds = time.time() - t
            return r, None, None
        r.n_matches = len(pa)
        Hf, inl = _homography(pa, pb, thr)
        r.n_inliers = inl
        r.inlier_ratio, r.cell_coverage = _match_set_quality(pa, pb, inl, ref_shape)
        r.seconds = time.time() - t
        if Hf is None:
            r.note = (r.note + "; " if r.note else "") + f"homography failed from {len(pa)} matches"
            return r, None, None
        total = Hf @ (Hc if coarse_ok else np.eye(3))
        evaluate(r, HomographyTransform(total), cps_tgt, cps_ref, px_per_mm, ref_shape)
        return r, (pa, pb), total

    r1, m1, _ = refined(M1, match_sift)
    results.append(r1)
    r2, m2, _ = refined(M2, match_orb)
    results.append(r2)
    r3, m3, _ = refined(M3, match_disk_lightglue)
    results.append(r3)

    # --- method 4: LoFTR, no coarse step, once per prior -----------------
    loftr: list[tuple[MethodResult, tuple | None]] = []
    for w in loftr_weights:
        t = time.time()
        r4 = MethodResult(loftr_name(w), "homography")
        m4 = None
        try:
            pa, pb = match_loftr(tgt, ref, pretrained=w)
            r4.n_matches = len(pa)
            H4, inl4 = _homography(pa, pb, 4.0)
            r4.n_inliers = inl4
            r4.inlier_ratio, r4.cell_coverage = _match_set_quality(pa, pb, inl4, ref_shape)
            if H4 is not None:
                evaluate(r4, HomographyTransform(H4), cps_tgt, cps_ref, px_per_mm, ref_shape)
                m4 = (pa, pb)
            else:
                r4.note = f"homography failed from {len(pa)} matches"
        except Exception as exc:  # noqa: BLE001
            r4.note = f"LoFTR ({w}) failed: {exc!r}"
        r4.seconds = time.time() - t
        results.append(r4)
        loftr.append((r4, m4))

    # --- method 5: TPS on the best method's matches ----------------------
    # The base matcher is chosen by MATCH-SET quality (coverage, inlier ratio,
    # inlier count) and never by control-point residual. See _pick_tps_base.
    pre_coarse = Hc if coarse_ok else np.eye(3)
    cand = [(r1, m1, pre_coarse), (r2, m2, pre_coarse), (r3, m3, pre_coarse)]
    cand += [(r, m, np.eye(3)) for r, m in loftr]

    r5 = MethodResult(M5, "tps")
    picked = _pick_tps_base(cand)
    if not picked:
        r5.note = "no matcher produced enough well-distributed inliers to build a TPS from"
    else:
        base, (pa, pb), pre = picked
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
            r5.n_matches, r5.n_inliers = len(pa_raw), len(sa)
            r5.inlier_ratio, r5.cell_coverage = _match_set_quality(sa, sb, len(sa), ref_shape)
            evaluate(r5, TPSPair(fwd, inv), cps_tgt, cps_ref, px_per_mm, ref_shape)
            # The NAME stays constant across pairs so pooling has a stable key
            # even when different pairs pick different bases. Which base was
            # picked goes in the note, ahead of anything evaluate() put there.
            r5.note = f"base: {base.name}" + (f"; {r5.note}" if r5.note else "")
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
# pooling - the gate statistic
# --------------------------------------------------------------------------

@dataclass
class DeadPair:
    """A pair that NO method could score. A data problem, not a method result."""

    target: str
    n_control_points: int
    why: str

    def summary(self) -> dict:
        return {"target": self.target, "n_control_points": self.n_control_points,
                "why": self.why}


def _why_dead(pair: dict) -> str:
    """Best available explanation for why nothing scored this pair."""
    n = pair["n_control_points"]
    if n < 3:
        return (f"only {n} control point{'' if n == 1 else 's'} shared with the "
                f"reference; scoring needs at least 3")
    notes: list[str] = []
    for r in pair["results"]:
        if r.note and r.note not in notes:
            notes.append(r.note)
    if notes:
        return "; ".join(notes[:2]) + (" (+more)" if len(notes) > 2 else "")
    return "no method produced an evaluable transform"


def partition_pairs(pairs: list[dict]) -> tuple[list[dict], list[DeadPair]]:
    """Split pairs into those at least one method scored, and those none did.

    A pair that NO method could score carries no information about any method.
    It usually means the control points were under-marked - fewer than three
    ids shared with the reference - which is an annotation problem, not a
    registration result.

    It has to be dropped before eligibility is judged, or it poisons the whole
    verdict: every method would be missing that pair, so every method would be
    incomplete, so nothing would be gate-eligible, and the headline would read
    NO-GO. That is a false NO-GO announcing a registration failure when one
    image was badly annotated - and the headline is the line that gets read at
    11pm on day 4, whatever the breakdown table says further down.

    Dropping is safe in the direction that matters: it can only ever ADD
    eligible methods, never excuse a method that genuinely failed on a pair
    other methods handled. A pair scored by SOME methods stays in the pool,
    and the methods that missed it stay ineligible.
    """
    live: list[dict] = []
    dead: list[DeadPair] = []
    for pair in pairs:
        if any(r.ok and r.n_residuals for r in pair["results"]):
            live.append(pair)
        else:
            dead.append(DeadPair(
                target=Path(pair["target"]).name,
                n_control_points=int(pair["n_control_points"]),
                why=_why_dead(pair),
            ))
    return live, dead


@dataclass
class Pooled:
    """One method's residuals from every pair, in one vector.

    This is what the gate reads. Per-pair numbers are a breakdown, not the
    statistic - see the module docstring for why median-of-medians is the
    wrong summary here.
    """

    name: str
    kind: str
    tier: int
    n: int                       # pooled residual count
    pairs_scored: int
    pairs_total: int
    median_mm: float = float("nan")
    p90_mm: float | None = None
    p95_mm: float | None = None
    max_mm: float = float("nan")
    mean_seconds: float = float("nan")
    median_matches: int = 0
    median_inliers: int = 0
    radial_pearson: float = float("nan")
    worst_pair: str = ""
    worst_pair_median_mm: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Scored on every pair. Required for gate eligibility."""
        return self.pairs_scored == self.pairs_total and self.pairs_total > 0

    def summary(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "tier": self.tier,
            "n_residuals": self.n, "pairs_scored": self.pairs_scored,
            "pairs_total": self.pairs_total, "complete": self.complete,
            "median_mm": self.median_mm, "p90_mm": self.p90_mm,
            "p95_mm": self.p95_mm, "max_mm": self.max_mm,
            "worst_pair": self.worst_pair,
            "worst_pair_median_mm": self.worst_pair_median_mm,
        }


def pct_or_none(v: np.ndarray, q: float) -> float | None:
    """Percentile `q`, or None if the sample is too small to mean it.

    On 10 values np.percentile(v, 95) interpolates between the 9th and 10th
    largest - it IS the maximum, give or take. Printing that next to a 3 mm
    gate threshold turns "p95 < 3 mm" into "worst control point < 3 mm", which
    is far harsher than intended and lets one poorly-marked mole fail the
    project. A number that cannot mean what it claims is refused rather than
    printed. See min_n_for_percentile.
    """
    if len(v) < min_n_for_percentile(q):
        return None
    return float(np.percentile(v, q))


def fmt_pct(value: float | None, n: int, q: float, unit: str = "") -> str:
    """A percentile, or an explicit refusal carrying the numbers.

    The unit attaches only to a real number, so a refusal never reads
    "n/a (n=12, need 20) mm".
    """
    if value is None:
        return f"n/a (n={n}, need {min_n_for_percentile(q)})"
    return f"{value:.2f}{unit}"


def pool_by_method(pairs: list[dict]) -> list[Pooled]:
    """Pool every residual per method, across every SCOREABLE pair. Ladder order.

    `pairs_total` counts only pairs that at least one method scored, so a
    single unscoreable pair cannot make every method incomplete at once. See
    partition_pairs.
    """
    live, _dead = partition_pairs(pairs)
    names: list[str] = []
    for p in pairs:
        for r in p["results"]:
            if r.name not in names:
                names.append(r.name)

    out: list[Pooled] = []
    for name in names:
        rows = [r for p in pairs for r in p["results"] if r.name == name]
        ok = [r for r in rows if r.ok and r.n_residuals]
        kind = ok[0].kind if ok else (rows[0].kind if rows else "-")
        pooled = Pooled(
            name=name,
            kind=kind,
            tier=method_tier(name),
            n=int(sum(r.n_residuals for r in ok)),
            pairs_scored=len(ok),
            pairs_total=len(live),
        )
        pooled.notes = [r.note for r in rows if r.note]
        if not ok:
            out.append(pooled)
            continue

        v = np.concatenate([r.residuals_mm for r in ok])
        pooled.median_mm = float(np.median(v))
        pooled.p90_mm = pct_or_none(v, 90)
        pooled.p95_mm = pct_or_none(v, 95)
        pooled.max_mm = float(v.max())
        pooled.mean_seconds = float(np.mean([r.seconds for r in rows]))
        pooled.median_matches = int(np.median([r.n_matches for r in ok]))
        pooled.median_inliers = int(np.median([r.n_inliers for r in ok]))
        rp = [r.radial_pearson for r in ok if not np.isnan(r.radial_pearson)]
        if rp:
            pooled.radial_pearson = float(np.median(rp))

        # The pair this method did worst on. Pooling is meant to stop a single
        # catastrophic pair hiding inside a median of medians, so name it.
        worst = max(ok, key=lambda r: r.median_mm)
        pooled.worst_pair = Path(
            [p["target"] for p in pairs for r in p["results"] if r is worst][0]
        ).name
        pooled.worst_pair_median_mm = worst.median_mm
        out.append(pooled)
    return out


@dataclass
class Verdict:
    status: str                  # GO | NO-GO | INCONCLUSIVE
    recommended: Pooled | None = None
    clearing: list[Pooled] = field(default_factory=list)
    inconclusive: list[Pooled] = field(default_factory=list)
    ineligible: list[Pooled] = field(default_factory=list)
    reason: str = ""


def decide_gate(pooled: list[Pooled]) -> Verdict:
    """Read the gate off the pooled statistics.

    Three outcomes, not two:

      GO            at least one method clears median AND p95 on the pool.
      INCONCLUSIVE  a method clears the median but p95 is unevaluable because
                    there are fewer than 20 pooled residuals. This is NOT a
                    GO. Declaring GO on the median alone while refusing to
                    print p95 would be having it both ways - either the sample
                    supports the gate's two halves or it does not.
      NO-GO         nothing clears.

    Only methods that scored on EVERY pair are eligible. A method missing a
    pair has a pool drawn from the pairs it happened to survive, which would
    reward it for failing on the hard one.
    """
    eligible = [m for m in pooled if m.complete and m.n]
    ineligible = [m for m in pooled if not (m.complete and m.n)]

    clears_median = [m for m in eligible if m.median_mm < config.GATE_MEDIAN_MM]
    clearing = [
        m for m in clears_median
        if m.p95_mm is not None and m.p95_mm < config.GATE_P95_MM
    ]
    inconclusive = [m for m in clears_median if m.p95_mm is None]

    if clearing:
        # Simplest tier first; pooled median only as an in-tier tiebreak.
        best = sorted(clearing, key=lambda m: (m.tier, m.median_mm, m.name))[0]
        return Verdict("GO", best, clearing, inconclusive, ineligible,
                       "simplest method clearing both halves of the gate")
    if inconclusive:
        best = sorted(inconclusive, key=lambda m: (m.tier, m.median_mm, m.name))[0]
        return Verdict("INCONCLUSIVE", best, [], inconclusive, ineligible,
                       "median clears but p95 has too few pooled residuals to evaluate")
    return Verdict("NO-GO", None, [], [], ineligible, "no method cleared the gate")


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

BIAS_NOTE = (
    "**The winning figure is optimistic.** It is the best of "
    "{n_methods} candidates selected on the same {n} control-point residuals it "
    "is reported against, and {n} residuals from {n_pairs} pair(s) of one "
    "subject is a small sample. The honest expectation on new photos is "
    "somewhat worse than the number above - treat the gate as cleared "
    "narrowly, not comfortably. Recommending the simplest method that clears, "
    "rather than the lowest number, limits how much of that optimism lands in "
    "the method actually chosen, but it does not remove it from the GO/NO-GO "
    "decision itself, which is still 'did any of {n_methods} clear'."
)


def write_report(pairs: list[dict], out: Path, plots: list[Path]) -> None:
    L: list[str] = ["# Registration bake-off - phase 0 question (b)\n"]

    pooled = pool_by_method(pairs)
    live, dead = partition_pairs(pairs)
    verdict = decide_gate(pooled)
    n_pairs = len(live)

    L.append("## VERDICT\n")

    # Loud, and above the verdict, because an unscoreable pair is an
    # annotation problem and would otherwise be read as a registration one.
    if dead:
        one = len(dead) == 1
        L.append(
            f"> **{len(dead)} pair{'' if one else 's'} could not be scored by ANY "
            f"method and {'is' if one else 'are'} excluded from the gate.** This is "
            f"a DATA problem, not a registration result - nothing below is a verdict "
            f"on {'that image' if one else 'those images'}.\n>"
        )
        for dp in dead:
            L.append(f"> - `{dp.target}` - {dp.why}")
        L.append(
            f">\n> Fix the annotation and re-run before reading anything into the "
            f"gate: mark more control points on "
            f"{'that image' if one else 'those images'} (and on the reference) with "
            f"`python -m src.controlpoints`. The gate below is judged against the "
            f"remaining {n_pairs} pair{'' if n_pairs == 1 else 's'}.\n"
        )

    L.append(
        f"- **Gate:** pooled median < {config.GATE_MEDIAN_MM} mm and pooled "
        f"p95 < {config.GATE_P95_MM} mm"
    )
    L.append(
        f"- **Statistic:** residuals from all {n_pairs} scoreable pair(s) pooled per "
        "method, not a median of per-pair medians. A per-pair median hides a single "
        "catastrophic pair, which is the failure this gate exists to catch."
    )

    if not any(m.n for m in pooled):
        L.append("\n**NO-GO - nothing scored.** No method produced a transform that could be "
                 "evaluated on any pair. This is almost certainly a data problem rather than "
                 "a registration one: check that control points exist for both images, that "
                 "at least three ids are SHARED between reference and target, and that the "
                 "fiducial is detected in the reference.\n")
    else:
        m = verdict.recommended
        L.append(f"\n**{verdict.status}.**\n")
        if m is not None:
            L.append(
                f"- **Recommended method:** `{m.name}` ({TIER_LABEL.get(m.tier, '?')}) - "
                f"pooled median **{m.median_mm:.2f} mm**, "
                f"p95 **{fmt_pct(m.p95_mm, m.n, 95, ' mm')}**, max {m.max_mm:.2f} mm, "
                f"n = {m.n} residuals over {m.pairs_scored} pair(s)"
            )
            L.append(f"- **Why this one:** {verdict.reason}.")
        if verdict.clearing:
            L.append(
                "- **Every method clearing the gate:** "
                + ", ".join(
                    f"`{c.name}` (median {c.median_mm:.2f}, p95 {fmt_pct(c.p95_mm, c.n, 95)}, n={c.n})"
                    for c in sorted(verdict.clearing, key=lambda c: (c.tier, c.median_mm))
                )
            )
        if verdict.inconclusive:
            L.append(
                "- **Clears the median but p95 is unevaluable:** "
                + ", ".join(f"`{c.name}` (n={c.n})" for c in verdict.inconclusive)
                + f". Needs {min_n_for_percentile(95)} pooled residuals; mark more "
                "control points or run more pairs."
            )
        if verdict.ineligible:
            L.append(
                "- **Not eligible for the gate** (did not score on every pair): "
                + ", ".join(
                    f"`{c.name}` ({c.pairs_scored}/{c.pairs_total})"
                    for c in verdict.ineligible
                )
                + ". A method that fails on the hardest pair would otherwise be "
                "scored only on the pairs it survived."
            )

        L.append("")
        if verdict.status == "GO":
            L.append(
                BIAS_NOTE.format(
                    n_methods=len([x for x in pooled if x.n]),
                    n=verdict.recommended.n,
                    n_pairs=verdict.recommended.pairs_scored,
                )
            )
            L.append(
                f"\nUse `{verdict.recommended.name}` for the phase 3 pipeline - but run the "
                "blink compare on it first. Summary statistics can look fine while the warp "
                "is wrong.\n"
            )
        elif verdict.status == "INCONCLUSIVE":
            L.append(
                "This is not a GO. The median half of the gate is cleared and the p95 half "
                "cannot be evaluated at this sample size. Do not advance the phase on half "
                "a gate; get to "
                f"{min_n_for_percentile(95)}+ pooled residuals first - more control points "
                "per pair, or more pairs.\n"
            )
        else:
            fails = []
            best_med = min((x.median_mm for x in pooled if x.n), default=float("nan"))
            if not (best_med < config.GATE_MEDIAN_MM):
                fails.append(f"best pooled median {best_med:.2f} mm >= {config.GATE_MEDIAN_MM} mm")
            withp95 = [x.p95_mm for x in pooled if x.n and x.p95_mm is not None]
            if withp95 and not (min(withp95) < config.GATE_P95_MM):
                fails.append(f"best pooled p95 {min(withp95):.2f} mm >= {config.GATE_P95_MM} mm")
            if fails:
                L.append("Failed on " + " and ".join(fails) + ".")
            L.append("\nCheapest next lever, in order:\n")
            L.append("1. Check the blink compare before believing any number here. If moles visibly "
                     "jitter, the warp is wrong in a way the summary statistics are hiding.")
            L.append("2. Look at the per-pair breakdown for one bad pair dragging the pool. A single "
                     "bad pair is a photo problem; uniformly bad pairs are a method problem.")
            L.append("3. If residuals are radially structured (see the correlation column), the "
                     "planar assumption is the problem - TPS is the fix, not a better matcher.")
            L.append("4. If inlier counts are low across every method, the problem is the photos: "
                     "depth of field, motion blur, or lighting change between sessions.")
            L.append("5. Only after those, reduce yaw to 50 deg and re-shoot.\n")

    # ---- pooled table -----------------------------------------------------
    L.append("\n## Pooled across all pairs - THIS IS WHAT THE GATE READS\n")
    L.append("Every control-point residual from every pair, in one vector per method. "
             "Errors in millimetres, converted through the reference image's own ArUco "
             "scale. `n` is the pooled residual count and is printed next to the "
             "percentiles because a percentile without its sample size is not "
             "interpretable.\n")
    L.append("| method | tier | n | pairs | median mm | p90 mm | p95 mm | max mm | worst pair (median) | matches | inliers | radial r | sec |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in pooled:
        if not m.n:
            note = m.notes[0] if m.notes else "no result"
            L.append(f"| {m.name} | - | 0 | 0/{m.pairs_total} | - | - | - | - | - | - | - | - | - |  <!-- {note} -->")
            continue
        flag = "" if m.complete else " !"
        L.append(
            f"| {m.name}{flag} | {TIER_LABEL.get(m.tier, '?')} | {m.n} | "
            f"{m.pairs_scored}/{m.pairs_total} | {m.median_mm:.2f} | "
            f"{fmt_pct(m.p90_mm, m.n, 90)} | {fmt_pct(m.p95_mm, m.n, 95)} | "
            f"{m.max_mm:.2f} | {m.worst_pair} ({m.worst_pair_median_mm:.2f}) | "
            f"{m.median_matches} | {m.median_inliers} | {m.radial_pearson:+.2f} | "
            f"{m.mean_seconds:.1f} |"
        )
    L.append("\n`!` marks a method that did not score on every pair; it is excluded from the "
             "gate. `matches` and `inliers` are per-pair medians - they do not pool.\n")

    bases = sorted({
        r.note for p in pairs for r in p["results"]
        if r.name == M5 and r.ok and r.note.startswith("base:")
    })
    if bases:
        L.append(
            "\nMethod 5 splines whichever matcher had the best-distributed match set on "
            "each pair, chosen without looking at the control points: "
            + "; ".join("`" + b.replace("base: ", "") + "`" for b in bases)
            + ". It pools under one name because it is one method.\n"
        )

    L.append("\n`radial r` is the Pearson correlation between residual magnitude and distance "
             "from the image centre. Strongly positive means the error grows toward the edges - "
             "the signature of fitting a plane to a face that is not one, and the evidence that "
             "a thin-plate spline is needed rather than a better feature matcher.\n")

    # ---- per-pair breakdown ----------------------------------------------
    L.append("\n## Per pair - breakdown only, not the gate\n")
    L.append("With ~10 control points per pair, a per-pair p90 or p95 is the top one or two "
             "residuals wearing a percentile's name. These columns are here to locate a bad "
             "pair, not to be compared against a threshold.\n")
    excluded = {dp.target for dp in dead}
    for p in pairs:
        name = Path(p["target"]).name
        flag = "  - **EXCLUDED FROM THE GATE**" if name in excluded else ""
        L.append(f"\n### {name} -> {Path(p['reference']).name}{flag}\n")
        L.append(f"- scale: **{p['px_per_mm']:.2f} px/mm** (reference image ArUco)")
        L.append(f"- control points: **{p['n_control_points']}** shared, held out")
        if name in excluded:
            L.append("- no method could be scored here, so this pair contributes "
                     "nothing to the pooled numbers and nothing to eligibility")
        L.append("")
        L.append("| method | n | median mm | p90 mm | p95 mm | max mm | matches | inliers | radial r | p | note |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in p["results"]:
            if not r.ok:
                L.append(f"| {r.name} | 0 | - | - | - | - | {r.n_matches} | {r.n_inliers} | - | - | {r.note} |")
                continue
            L.append(
                f"| {r.name} | {r.n_residuals} | {r.median_mm:.2f} | {r.p90_mm:.2f} | "
                f"{r.p95_mm:.2f} | {r.max_mm:.2f} | {r.n_matches} | {r.n_inliers} | "
                f"{r.radial_pearson:+.2f} | {r.radial_p:.3f} | {r.note or 'ok'} |"
            )

    if plots:
        L.append("\n## Residual vector plots\n")
        for p in plots:
            L.append(f"- `{p.relative_to(config.REPO) if config.REPO in p.parents else p}`")

    L.append("\n## Read this before trusting the table\n")
    L.append("- Control points were used for SCORING only. No transform in this report was "
             "fitted using them, and - unlike an earlier version of this file - method 5 no "
             "longer uses them to pick which matcher to spline either. Its base is chosen by "
             "match-set coverage and inlier ratio, which are properties of the matches rather "
             "than of the held-out truth.")
    L.append("- The gate reads the POOLED row, never a per-pair row.")
    L.append("- The recommendation is the simplest method that clears, not the lowest number. "
             "A method that wins by 0.05 mm on 50 points has not earned its extra machinery.")
    L.append("- Summary statistics can look fine while the warp is subtly wrong. Run the blink "
             "compare on the recommended method before declaring the gate passed:\n")
    L.append("  ```\n  python -m src.register review --ref REF.jpg --target T.jpg --method tps\n  ```")
    L.append("- For a TPS the blink compare shows a Newton-refined inverse of the same spline "
             "that produced the numbers, and the review screen prints the measured round-trip "
             "error so you can see how far the picture is from the map being scored.\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# review / blink compare
# --------------------------------------------------------------------------

def review(
    ref_path: Path,
    tgt_path: Path,
    method: str = "tps",
    hz: float = 2.0,
    loftr_weights: Sequence[str] = LOFTR_WEIGHTS,
) -> int:
    """Side-by-side plus blink compare. The honest check on a warp.

    The readout names the round-trip error of the map used to BUILD THE
    PICTURE against the map that produced the NUMBERS. For a homography that
    is exact - the inverse is algebraic. For a TPS it is measured, because the
    dense warp has to invert the spline numerically, and a blink compare
    showing a different transform than the one being scored would defeat the
    entire purpose of looking.
    """
    pair = run_pair(ref_path, tgt_path, loftr_weights=loftr_weights)
    ok = [r for r in pair["results"] if r.ok]
    if not ok:
        print("no method produced an evaluable transform")
        return 1

    want = [r for r in ok if method.lower() in r.name.lower() or method.lower() == r.kind]
    r = (want or sorted(ok, key=lambda x: (method_tier(x.name), x.median_mm)))[0]
    print(f"reviewing: {r.name}" + (f"   ({r.note})" if r.note else ""))

    ref = cv2.imread(str(ref_path))
    tgt = cv2.imread(str(tgt_path))
    warped = r.transform.warp_to_reference(tgt, pair["ref_shape"])

    rt = getattr(r.transform, "last_roundtrip", None)
    if rt is None:
        rt_text = "round-trip exact (analytic inverse)"
    else:
        ppm = pair["px_per_mm"]
        rt_text = (
            f"round-trip max {rt['max_px'] / ppm:.4f} mm "
            f"(in hull {rt['max_in_hull_px'] / ppm:.4f} mm; "
            f"unrefined twin fit was {rt['seed_max_px'] / ppm:.4f} mm, "
            f"{rt['newton_iters']} Newton it)"
        )
    print(f"  {rt_text}")
    print("  This is how far the PICTURE below is from the map the millimetres score.")

    fm_ok = "OK" if any(m.name.startswith("0") and m.ok for m in pair["results"]) else "FAILED"
    readout = (
        f"median {r.median_mm:.2f} mm | p95 {r.p95_mm:.2f} mm (n={r.n_residuals}) | "
        f"matches {r.n_inliers} | {r.kind} | {pair['px_per_mm']:.2f} px/mm | "
        f"FaceMesh {fm_ok} | {rt_text}"
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
    cv2.putText(bar, readout, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    sbs_small = np.vstack([sbs_small, bar])

    sb = scr_w // 2
    sa_ = sb / ref.shape[1]
    blink = [cv2.resize(a, None, fx=sa_, fy=sa_), cv2.resize(b, None, fx=sa_, fy=sa_)]
    bbar = np.zeros((46, blink[0].shape[1], 3), np.uint8)
    cv2.putText(bbar, readout, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
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
    b.add_argument("--loftr-weights", nargs="+", default=list(LOFTR_WEIGHTS),
                   choices=["indoor", "outdoor"],
                   help="LoFTR priors to run; each is a separate ladder entry")

    r = sub.add_parser("review")
    r.add_argument("--ref", type=Path, required=True)
    r.add_argument("--target", type=Path, required=True)
    r.add_argument("--method", default="tps")
    r.add_argument("--hz", type=float, default=2.0)
    r.add_argument("--cp-dir", type=Path, default=None)
    r.add_argument("--loftr-weights", nargs="+", default=list(LOFTR_WEIGHTS),
                   choices=["indoor", "outdoor"])

    a = ap.parse_args(argv)
    if getattr(a, "cp_dir", None):
        set_controlpoints_dir(a.cp_dir)
    if a.cmd == "review":
        try:
            return review(a.ref, a.target, a.method, a.hz, a.loftr_weights)
        except RegistrationInputError as exc:
            print(f"cannot review this pair:\n{exc}")
            return 3

    tgts = [p for p in _collect(a.targets) if p.resolve() != a.ref.resolve()]
    if not tgts:
        print("no target images found")
        return 2

    pairs, plots, skipped = [], [], []
    for t in tgts:
        print(f"\n=== {t.name} -> {a.ref.name} ===")
        try:
            p = run_pair(a.ref, t, a.tps_lambda, loftr_weights=a.loftr_weights)
        except RegistrationInputError as exc:
            # A bad pair must not take the batch down, and must not vanish
            # either: it is listed again at the end.
            print(f"  SKIPPED - {exc}")
            skipped.append((t, str(exc)))
            continue
        for res in p["results"]:
            if res.ok:
                print(f"  {res.name:34s} median {res.median_mm:6.2f} mm   "
                      f"p95 {res.p95_mm:6.2f} mm (n={res.n_residuals:2d})   "
                      f"inliers {res.n_inliers:5d}   {res.seconds:5.1f}s")
            else:
                print(f"  {res.name:34s} FAILED  {res.note}")
        pairs.append(p)
        pl = plot_residuals(p, a.plots)
        if pl:
            plots.append(pl)

    if skipped:
        print("\nSKIPPED PAIRS (not in the report, not in the pool):")
        for t, why in skipped:
            print(f"  {t.name}: {why.splitlines()[0]}")
    if not pairs:
        print("\nno pair could be scored - nothing to report")
        return 3

    write_report(pairs, a.report, plots)

    live, dead = partition_pairs(pairs)
    if dead:
        print("\nPAIRS SCORED BY NO METHOD - excluded from the gate (DATA problem):")
        for dp in dead:
            print(f"  {dp.target}: {dp.why}")
        print(f"  the gate below is judged against the remaining {len(live)} "
              f"pair{'' if len(live) == 1 else 's'}")

    print("\nPOOLED (what the gate reads):")
    pooled = pool_by_method(pairs)
    for m in pooled:
        if not m.n:
            continue
        print(f"  {m.name:34s} median {m.median_mm:6.2f} mm   "
              f"p95 {fmt_pct(m.p95_mm, m.n, 95):>22s}   n={m.n:4d}   "
              f"pairs {m.pairs_scored}/{m.pairs_total}")
    v = decide_gate(pooled)
    print(f"\n{v.status}" + (f" - recommend {v.recommended.name} ({v.reason})"
                              if v.recommended else ""))
    j = a.report.with_suffix(".json")
    j.write_text(
        json.dumps(
            {
                "gate": {
                    "median_mm": config.GATE_MEDIAN_MM,
                    "p95_mm": config.GATE_P95_MM,
                    "statistic": "residuals pooled across all pairs, per method",
                },
                "verdict": {
                    "status": v.status,
                    "recommended": v.recommended.name if v.recommended else None,
                    "reason": v.reason,
                    "rule": "simplest complexity tier that clears; pooled median "
                            "only as an in-tier tiebreak",
                    "bias": "selected best-of-N on a small single-subject sample; "
                            "the honest read on new data is somewhat worse",
                },
                "pooled": [m.summary() for m in pooled],
                "excluded_pairs": [dp.summary() for dp in dead],
                "skipped_pairs": [{"target": str(t), "why": why} for t, why in skipped],
                "pairs": [
                    {
                        "reference": str(p["reference"]), "target": str(p["target"]),
                        "px_per_mm": p["px_per_mm"],
                        "n_control_points": p["n_control_points"],
                        "results": [r.summary() for r in p["results"]],
                    }
                    for p in pairs
                ],
            },
            indent=2, default=float,
        ),
        encoding="utf-8",
    )
    print(f"wrote {j}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
