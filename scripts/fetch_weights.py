"""Pre-fetch every model weight the phase 0 spike needs, into models/weights/.

Run this once while you have network. After it completes the spike is
fully offline-capable: nothing in facemesh_check.py, register.py or the
server reaches the internet.

Caches:
  models/weights/face_landmarker.task     MediaPipe Tasks face landmarker (478 pts)
  models/weights/hub/checkpoints/*.ckpt   kornia: LoFTR, DISK, LightGlue
  tests/fixtures/portrait.jpg             one public sample face for smoke tests
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = REPO / "models" / "weights"
FIXTURES = REPO / "tests" / "fixtures"

# kornia/torch download into TORCH_HOME/hub/checkpoints. Point it in-repo so
# the weights are version-pinned next to the code rather than in a user cache
# that a Windows reset would wipe.
os.environ["TORCH_HOME"] = str(WEIGHTS)

DIRECT_DOWNLOADS = [
    (
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task",
        WEIGHTS,
    ),
    # A single public sample face, used only to prove the code paths run
    # before Noah's real photos exist. Never enters data/.
    (
        "portrait.jpg",
        "https://storage.googleapis.com/mediapipe-assets/portrait.jpg",
        FIXTURES,
    ),
]


def _curl(url: str, dest: Path) -> bool:
    """Fall back to Windows curl.exe, which validates against the Windows
    certificate store rather than Python's certifi bundle.

    cmp.felk.cvut.cz (the LoFTR host) serves a chain certifi cannot complete,
    so urllib dies with CERTIFICATE_VERIFY_FAILED while curl.exe succeeds.
    """
    try:
        r = subprocess.run(
            ["curl.exe", "-sSL", "--fail", "-o", str(dest), url],
            capture_output=True,
            text=True,
            timeout=900,
        )
        return r.returncode == 0 and dest.exists() and dest.stat().st_size > 0
    except Exception:  # noqa: BLE001
        return False


def fetch(name: str, url: str, dest_dir: Path) -> bool:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cached] {name}  ({dest.stat().st_size/1024:.0f} KiB)")
        return True
    print(f"  [get]    {name}  <- {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "lesion-atlas/p0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn]   urllib failed ({type(exc).__name__}); trying curl.exe")
        if dest.exists():
            dest.unlink()
        if not _curl(url, dest):
            print(f"  [FAIL]   {name}: {exc}")
            if dest.exists():
                dest.unlink()
            return False
    print(f"  [ok]     {name}  ({dest.stat().st_size/1024:.0f} KiB)")
    return True


# torch.hub names its cache file after the URL basename. Pre-seeding these by
# hand means the kornia constructors below find them already cached and never
# hit the unverifiable host at all.
HUB_PRESEED = [
    ("loftr_outdoor.ckpt", "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt"),
    ("loftr_indoor.ckpt", "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor.ckpt"),
]


def main() -> int:
    ok = True
    print("=" * 68)
    print("direct downloads")
    print("=" * 68)
    for name, url, dest in DIRECT_DOWNLOADS:
        ok &= fetch(name, url, dest)

    print("=" * 68)
    print("kornia / torch.hub weights  (TORCH_HOME=%s)" % WEIGHTS)
    print("=" * 68)
    ckpt_dir = WEIGHTS / "hub" / "checkpoints"
    for name, url in HUB_PRESEED:
        ok &= fetch(name, url, ckpt_dir)
    try:
        import torch  # noqa: F401
        from kornia.feature import DISK, LightGlue, LoFTR

        print("  [get]    LoFTR(pretrained='outdoor')")
        LoFTR(pretrained="outdoor")
        print("  [ok]     LoFTR outdoor")

        print("  [get]    LoFTR(pretrained='indoor')")
        LoFTR(pretrained="indoor")
        print("  [ok]     LoFTR indoor")

        print("  [get]    DISK.from_pretrained('depth')")
        DISK.from_pretrained("depth")
        print("  [ok]     DISK depth")

        print("  [get]    LightGlue('disk')")
        LightGlue("disk")
        print("  [ok]     LightGlue disk")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL]   kornia weights: {exc!r}")
        ok = False

    print("=" * 68)
    cached = sorted(p for p in WEIGHTS.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in cached)
    for p in cached:
        print(f"  {p.relative_to(REPO)}  {p.stat().st_size/1024/1024:.1f} MiB")
    print(f"  TOTAL {total/1024/1024:.1f} MiB")
    print("VERDICT:", "PASS - all weights cached" if ok else "FAIL - see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
