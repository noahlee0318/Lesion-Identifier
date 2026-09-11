"""Did iOS Safari alter the file on its way through the upload page?

Run this ONCE before the first real session. It decides whether the upload
path is usable for a longitudinal series at all.

  1. Shoot one photo with the Camera app.
  2. Copy it to the laptop over USB (or AirDrop) -> that is the CONTROL.
  3. Upload the SAME photo through the upload page -> that is the SUBJECT.
  4. Point this script at both.

It compares dimensions, byte size, SHA-256 and every EXIF field, and prints a
verdict. If Safari re-encoded, resized or stripped metadata, this says so
loudly and names exactly what changed.

Why it matters: a re-encode changes pixel values. Lesion detection at 1.5 mm
runs on a handful of pixels, and JPEG generation loss lands precisely on the
high-frequency detail that distinguishes a papule from noise. A resize is
worse - it silently changes px/mm and every millimetre downstream.

    python scripts/verify_upload_fidelity.py --usb IMG_4821.JPG --uploaded left60_main_iphone15_070311.jpg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.server import qa  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(p: Path) -> dict:
    data = p.read_bytes()
    exif = qa.read_exif(data)
    q = qa.check(data, require_fiducial=False)
    return {
        "path": str(p),
        "name": p.name,
        "bytes": len(data),
        "sha256": sha256(p),
        "width_px": q.width_px,
        "height_px": q.height_px,
        "exif": exif,
        "lap_var": q.lap_var,
        "mean_luma": q.mean_luma,
    }


# EXIF tags that SHOULD differ or are harmless noise.
IGNORABLE = {"pil_size", "pil_format"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usb", type=Path, required=True, help="file copied over USB (control)")
    ap.add_argument("--uploaded", type=Path, required=True, help="file stored by the server (subject)")
    ap.add_argument("--json", type=Path, default=None, help="also write the full diff here")
    a = ap.parse_args()

    for p in (a.usb, a.uploaded):
        if not p.exists():
            print(f"missing file: {p}")
            return 2

    A, B = describe(a.usb), describe(a.uploaded)

    print("=" * 74)
    print("UPLOAD FIDELITY CHECK")
    print("=" * 74)
    print(f"  control (USB)   {A['name']}")
    print(f"  subject (HTTP)  {B['name']}")
    print("-" * 74)

    identical = A["sha256"] == B["sha256"]
    same_dims = (A["width_px"], A["height_px"]) == (B["width_px"], B["height_px"])
    same_size = A["bytes"] == B["bytes"]

    def line(label, va, vb, ok):
        flag = "same" if ok else ">>> DIFFERS"
        print(f"  {label:<16} {str(va):<28} {str(vb):<28} {flag}")

    print(f"  {'':<16} {'CONTROL':<28} {'SUBJECT':<28}")
    line("dimensions", f"{A['width_px']}x{A['height_px']}", f"{B['width_px']}x{B['height_px']}", same_dims)
    line("bytes", f"{A['bytes']:,}", f"{B['bytes']:,}", same_size)
    line("sha256", A["sha256"][:16] + "...", B["sha256"][:16] + "...", identical)
    line("laplacian var", f"{A['lap_var']:.1f}", f"{B['lap_var']:.1f}",
         abs(A["lap_var"] - B["lap_var"]) < 1e-6)

    # --- EXIF diff -----------------------------------------------------
    ka, kb = set(A["exif"]) - IGNORABLE, set(B["exif"]) - IGNORABLE
    lost, gained = sorted(ka - kb), sorted(kb - ka)
    changed = sorted(k for k in (ka & kb) if A["exif"][k] != B["exif"][k])

    print("-" * 74)
    print(f"  EXIF: {len(ka)} tags in control, {len(kb)} in subject")
    if lost:
        print(f"  STRIPPED ({len(lost)}): {', '.join(lost[:14])}"
              + (f" ... +{len(lost)-14} more" if len(lost) > 14 else ""))
    if gained:
        print(f"  ADDED ({len(gained)}): {', '.join(gained[:14])}")
    if changed:
        print(f"  CHANGED ({len(changed)}):")
        for k in changed[:14]:
            print(f"      {k}: {str(A['exif'][k])[:32]!r} -> {str(B['exif'][k])[:32]!r}")
    if not (lost or gained or changed):
        print("  no EXIF differences")

    # --- verdict -------------------------------------------------------
    print("=" * 74)
    if identical:
        print("VERDICT: PASS - byte-for-byte identical.")
        print("  iOS Safari passed the file through untouched. The upload path is")
        print("  safe for the longitudinal series. Use it.")
        rc = 0
    elif same_dims and not lost and not changed:
        print("VERDICT: MARGINAL - bytes differ but pixels and metadata survived.")
        print("  Dimensions and EXIF match, so this is most likely a container")
        print("  rewrite rather than a re-encode. Check the laplacian variance above:")
        print("  if it moved at all, the pixels were re-compressed and this is a FAIL.")
        rc = 0
    else:
        print("VERDICT: FAIL - Safari altered the file.")
        if not same_dims:
            print(f"  RESIZED: {A['width_px']}x{A['height_px']} -> {B['width_px']}x{B['height_px']}")
            print("    This is the serious one. A resize changes px/mm, so every")
            print("    millimetre the project reports would be wrong, and lesion")
            print("    detail at the 1.5 mm threshold is destroyed outright.")
        if lost:
            print(f"  STRIPPED {len(lost)} EXIF tags, including: {', '.join(lost[:6])}")
        if changed:
            print(f"  REWROTE {len(changed)} EXIF tags.")
        print()
        print("  DO NOT run the longitudinal series through this path as-is.")
        print("  Options, cheapest first:")
        print("   1. Photos -> Transfer to Mac or PC -> Keep Originals, and Camera ->")
        print("      Formats -> Most Compatible. Re-run this check.")
        print("   2. In the file picker choose 'Browse' / Files rather than Photo")
        print("      Library - the Files path does not transcode.")
        print("   3. If neither works, the upload page stays for QA and covariates")
        print("      and the images come over by USB. That costs a daily step, which")
        print("      is exactly the friction that kills adherence - so exhaust 1 and 2.")
        rc = 1
    print("=" * 74)

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(
            json.dumps(
                {"control": A, "subject": B,
                 "diff": {"stripped": lost, "added": gained, "changed": changed},
                 "identical": identical, "same_dimensions": same_dims},
                indent=2, default=str,
            ),
            encoding="utf-8",
        )
        print(f"wrote {a.json}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
