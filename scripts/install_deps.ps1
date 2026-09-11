# Lesion Atlas - dependency install (phase 0)
# Run once. Installs torch+torchvision from the CUDA wheel index, then the rest.
# Python 3.12 is REQUIRED: mediapipe publishes no 3.13/3.14 wheels.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py   = Join-Path $repo ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) { throw "venv missing at $py - create it with python3.12 -m venv .venv" }

Write-Output "=== interpreter ==="
& $py --version

Write-Output "=== 1/3 torch + torchvision (CUDA 12.8) ==="
& $py -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128

Write-Output "=== 2/3 cv + mediapipe + science stack ==="
# opencv-contrib-python ONLY. Never install opencv-python alongside it:
# the two ship the same cv2 module and whichever lands last wins, which
# silently removes the ArUco module we depend on.
& $py -m pip install --upgrade `
    opencv-contrib-python `
    mediapipe `
    numpy scipy matplotlib pillow `
    kornia kornia-rs `
    fastapi "uvicorn[standard]" python-multipart jinja2 `
    qrcode `
    piexif

Write-Output "=== 3/3 verify ==="
& $py (Join-Path $repo "scripts\verify_env.py")
