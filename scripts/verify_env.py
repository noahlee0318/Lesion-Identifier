"""Environment verification for the phase 0 spike.

Prints a blunt PASS/FAIL per requirement. Exits non-zero if anything
load-bearing is missing, so install_deps.ps1 fails loudly rather than
leaving a half-working venv behind.

The CUDA check is deliberately not forgiving: CLAUDE.md says say so plainly
if CUDA is not available rather than silently falling back to CPU.
"""

from __future__ import annotations

import importlib
import platform
import sys

RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((label, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    print("=" * 70)
    print("Lesion Atlas - phase 0 environment check")
    print("=" * 70)
    print(f"python   {sys.version}")
    print(f"platform {platform.platform()}")
    print(f"exe      {sys.executable}")
    print("-" * 70)

    major_minor = sys.version_info[:2]
    check(
        "Python 3.12.x",
        major_minor == (3, 12),
        f"found {major_minor[0]}.{major_minor[1]} (mediapipe has no 3.13/3.14 wheels)",
    )

    # --- plain imports -------------------------------------------------
    for mod in ("numpy", "scipy", "matplotlib", "PIL", "fastapi", "uvicorn", "qrcode"):
        try:
            m = importlib.import_module(mod)
            check(f"import {mod}", True, getattr(m, "__version__", "?"))
        except Exception as exc:  # noqa: BLE001 - we want the message verbatim
            check(f"import {mod}", False, repr(exc))

    # --- opencv, and specifically the contrib ArUco module -------------
    try:
        import cv2

        check("import cv2", True, cv2.__version__)
        has_aruco = hasattr(cv2, "aruco")
        check(
            "cv2.aruco (contrib build)",
            has_aruco,
            "present" if has_aruco else "MISSING - you have plain opencv-python, not contrib",
        )
        if has_aruco:
            d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            check("DICT_4X4_50 dictionary", d is not None)
    except Exception as exc:  # noqa: BLE001
        check("import cv2", False, repr(exc))

    # Two cv2 distributions in one venv is a silent footgun - detect it.
    try:
        from importlib.metadata import distributions

        cv_dists = sorted(
            d.metadata["Name"]
            for d in distributions()
            if (d.metadata["Name"] or "").lower().startswith("opencv")
        )
        check(
            "exactly one opencv distribution",
            cv_dists == ["opencv-contrib-python"],
            f"found {cv_dists}",
        )
    except Exception as exc:  # noqa: BLE001
        check("opencv distribution scan", False, repr(exc))

    # --- mediapipe ------------------------------------------------------
    # mediapipe 1.0 removed `mp.solutions` entirely, so the legacy
    # FaceMesh(refine_landmarks=True) no longer exists. The Tasks API
    # FaceLandmarker is the replacement and emits the same 478 landmarks.
    try:
        import mediapipe as mp

        check("import mediapipe", True, mp.__version__)
        legacy = hasattr(mp, "solutions")
        check(
            "mediapipe API flavour",
            True,
            "legacy mp.solutions present" if legacy else "1.0+ Tasks API (mp.solutions removed)",
        )

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        check("mediapipe.tasks.vision.FaceLandmarker", hasattr(vision, "FaceLandmarker"))

        from pathlib import Path as _P

        task = _P(__file__).resolve().parent.parent / "models" / "weights" / "face_landmarker.task"
        if not task.exists():
            check(
                "face_landmarker.task cached",
                False,
                f"missing {task} - run scripts/fetch_weights.py",
            )
        else:
            check("face_landmarker.task cached", True, f"{task.stat().st_size/1024/1024:.1f} MiB")
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(task)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                output_facial_transformation_matrixes=True,
            )
            lm = vision.FaceLandmarker.create_from_options(opts)
            lm.close()
            check("FaceLandmarker constructs (478-landmark bundle)", True)
    except Exception as exc:  # noqa: BLE001
        check("mediapipe Tasks FaceLandmarker", False, repr(exc))

    # --- torch + CUDA ---------------------------------------------------
    try:
        import torch

        check("import torch", True, torch.__version__)
        cuda_ok = torch.cuda.is_available()
        detail = ""
        if cuda_ok:
            detail = (
                f"{torch.cuda.get_device_name(0)}  "
                f"cuda={torch.version.cuda}  "
                f"vram={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB"
            )
            # Actually move a tensor - is_available() has lied before.
            x = torch.randn(256, 256, device="cuda")
            _ = (x @ x).sum().item()
            torch.cuda.synchronize()
        else:
            detail = "torch.cuda.is_available() is False - this is a CPU-only build or a driver problem"
        check("torch.cuda.is_available()", cuda_ok, detail)

        import torchvision

        check("import torchvision", True, torchvision.__version__)
    except Exception as exc:  # noqa: BLE001
        check("import torch", False, repr(exc))

    # --- kornia (LightGlue/DISK/LoFTR live here) ------------------------
    try:
        import kornia

        check("import kornia", True, kornia.__version__)
        from kornia.feature import DISK, LightGlue, LoFTR  # noqa: F401

        check("kornia.feature DISK/LightGlue/LoFTR symbols", True)
    except Exception as exc:  # noqa: BLE001
        check("import kornia", False, repr(exc))

    try:
        import kornia_rs

        check("import kornia_rs", True, getattr(kornia_rs, "__version__", "?"))
    except Exception as exc:  # noqa: BLE001
        check("import kornia_rs", False, repr(exc))

    print("-" * 70)
    failed = [label for label, ok, _ in RESULTS if not ok]
    if failed:
        print(f"VERDICT: FAIL - {len(failed)} check(s) failed:")
        for label in failed:
            print(f"  - {label}")
        return 1
    print(f"VERDICT: PASS - all {len(RESULTS)} checks green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
