"""Phase 0 ingest server. Plain HTTP, LAN only, no camera code.

A file input needs no secure context, so this deliberately runs over plain
HTTP. HTTPS arrives in phase 1 with getUserMedia, not before.

  GET  /           upload page (iPhone-sized)
  GET  /probe      throwaway getUserMedia resolution probe (captures nothing)
  GET  /health     liveness, for the page's "server unreachable" state
  GET  /calibration  phase 0 calibration progress
  POST /capture    store bytes UNMODIFIED + sidecar JSON + SQLite rows
  POST /regimen    log a regimen change

    python -m src.server.main
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import socket
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse

try:
    from .. import config
    from . import db, qa
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src import config
    from src.server import db, qa

STATIC = Path(__file__).parent / "static"
SIDECAR_SCHEMA_VERSION = 1
PAGE_VERSION = "p0-ingest-1"

log = logging.getLogger("lesion_atlas")


def setup_logging(verbose: bool = False) -> None:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.handlers.clear()
    # Rotating, because a failure at 7am has to still be diagnosable at 7pm.
    fh = RotatingFileHandler(
        config.LOGS_DIR / "server.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    log.addHandler(sh)


app = FastAPI(title="Lesion Atlas ingest", version=PAGE_VERSION)
_con = None


def con():
    global _con
    if _con is None:
        config.ensure_dirs()
        _con = db.connect()
    return _con


# --------------------------------------------------------------------------

@app.get("/health")
def health():
    try:
        c = con()
        n = c.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"]
        return {
            "ok": True,
            "time": datetime.now().isoformat(timespec="seconds"),
            "data_root": str(config.DATA_ROOT),
            "images": n,
            "version": PAGE_VERSION,
        }
    except Exception as exc:  # noqa: BLE001
        log.exception("health check failed")
        return JSONResponse({"ok": False, "error": repr(exc)}, status_code=500)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/probe")
def probe():
    return FileResponse(STATIC / "probe.html")


@app.post("/probe/measure")
async def probe_measure(file: UploadFile = File(...)):
    """Measure px/mm on a frame grabbed from the video stream.

    Deliberately does NOT store anything. The probe exists to size the stream,
    and a canvas grab off a preview is not a dataset image - letting one reach
    data/raw would quietly contaminate the series with a different optical path.
    """
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty frame")
    q = qa.check(data, require_fiducial=False)
    return {
        "aruco_found": q.aruco_found,
        "px_per_mm": q.aruco_px_per_mm,
        "lesion_px_at_1p5mm": q.lesion_px_at_1p5mm,
        "edge_spread_pct": q.aruco_edge_spread_pct,
        "width_px": q.width_px,
        "height_px": q.height_px,
        "lap_var": q.lap_var,
        "stored": False,
        "note": "" if q.aruco_found else "no ArUco marker detected in the grabbed frame",
    }


@app.get("/calibration")
def calibration():
    return db.calibration_progress(con())


@app.get("/recent")
def recent():
    return {"sessions": db.recent_sessions(con()), "regimen": db.recent_regimen(con())}


# --------------------------------------------------------------------------

@app.post("/capture")
async def capture(
    file: UploadFile = File(...),
    pose: str = Form(...),
    lens: str = Form("main"),
    device: str = Form("iphone15"),
    session_date: str = Form(...),
    kind: str = Form("session"),
    covariates: str = Form("{}"),
    client: str = Form("{}"),
):
    if pose not in config.POSES:
        raise HTTPException(400, f"pose must be one of {config.POSES}")
    if lens not in config.LENSES:
        raise HTTPException(400, f"lens must be one of {config.LENSES}")
    if kind not in ("session", "calib"):
        raise HTTPException(400, "kind must be 'session' or 'calib'")
    try:
        datetime.strptime(session_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, "session_date must be YYYY-MM-DD")

    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")

    # ---- store the received bytes, unmodified -------------------------
    # No decode-re-encode, no resize, no EXIF strip. The QA pass below
    # decodes a COPY in memory and never touches what goes to disk.
    sha = qa.sha256_of(data)
    dup = db.find_by_sha(con(), sha)
    if dup:
        log.info("duplicate upload sha=%s already at %s", sha[:12], dup["path"])
        return {
            "ok": True, "duplicate": True, "pose": pose, "lens": lens,
            "stored_path": dup["path"], "sha256": sha,
            "qa": json.loads(dup["qa_json"] or "{}"),
            "message": "identical bytes already stored - nothing written",
        }

    now = datetime.now()
    day_dir = (config.CALIB_DIR if kind == "calib" else config.RAW_DIR / session_date)
    day_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{pose}_{lens}_{device}_{now.strftime('%H%M%S')}"
    if kind == "calib":
        stem = f"{session_date}_{stem}"
    dest = day_dir / f"{stem}.jpg"
    i = 1
    while dest.exists():
        dest = day_dir / f"{stem}_{i}.jpg"
        i += 1
    dest.write_bytes(data)

    # ---- measure (on a copy) ------------------------------------------
    q = qa.check(data)
    exif = qa.read_exif(data)

    try:
        cov = json.loads(covariates) or {}
    except json.JSONDecodeError:
        cov = {}
    try:
        cli = json.loads(client) or {}
    except json.JSONDecodeError:
        cli = {}

    sidecar = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "capture": {
            "session_date": session_date,
            "pose": pose,
            "lens": lens,
            "device": device,
            "kind": kind,
            "received_at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "received_at_local": now.astimezone().isoformat(timespec="milliseconds"),
        },
        "file": {
            "stored_path": str(dest),
            "original_filename": file.filename,
            "bytes": len(data),
            "sha256": sha,
            "content_type": file.content_type,
            "width_px": q.width_px,
            "height_px": q.height_px,
        },
        "client": {
            "user_agent": cli.get("user_agent"),
            "file_last_modified": cli.get("file_last_modified"),
            "timezone": cli.get("timezone"),
            "page_version": cli.get("page_version", PAGE_VERSION),
        },
        "exif": exif,
        "covariates": cov,
        "qa": q.to_dict(),
    }
    side = dest.with_suffix(".json")
    side.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    with db.tx(con()) as c:
        sid = db.get_or_create_session(
            c, session_date, now.isoformat(timespec="seconds"), device, cov, kind
        )
        db.insert_image(
            c, sid, pose=pose, lens=lens, device=device, path=str(dest),
            sidecar_path=str(side), original_filename=file.filename, bytes=len(data),
            sha256=sha, width_px=q.width_px, height_px=q.height_px,
            exif_json=json.dumps(exif), qa_json=json.dumps(q.to_dict()),
            qa_verdict=q.verdict,
        )

    log.info(
        "stored %s  %s/%s  %d bytes  sha=%s  QA=%s  %s",
        dest.name, pose, lens, len(data), sha[:12], q.verdict, "; ".join(q.reasons or []),
    )
    return {
        "ok": True, "duplicate": False, "pose": pose, "lens": lens,
        "stored_path": str(dest), "sha256": sha, "bytes": len(data),
        "width_px": q.width_px, "height_px": q.height_px,
        "qa": q.to_dict(),
        "calibration": db.calibration_progress(con()) if kind == "calib" else None,
    }


@app.post("/regimen")
def regimen(
    event_date: str = Form(...),
    change: str = Form(...),
    product: str = Form(...),
    detail: str = Form(""),
):
    if change not in ("added", "stopped", "dose_changed"):
        raise HTTPException(400, "change must be added | stopped | dose_changed")
    if not product.strip():
        raise HTTPException(400, "product is required")
    with db.tx(con()) as c:
        rid = db.add_regimen_event(c, event_date, change, product.strip(), detail.strip() or None)
    log.info("regimen event %s %s %s", event_date, change, product)
    return {"ok": True, "id": rid, "recent": db.recent_regimen(con())}


# --------------------------------------------------------------------------

def lan_ip() -> str:
    """Best guess at the address the phone should use."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))          # no packet is sent
        return s.getsockname()[0]
    except Exception:  # noqa: BLE001
        return "127.0.0.1"
    finally:
        s.close()


def print_banner(host: str, port: int) -> None:
    url = f"http://{lan_ip()}:{port}/"
    print()
    print("=" * 62)
    print("  LESION ATLAS - phase 0 ingest server")
    print("=" * 62)
    print(f"  on your phone:  {url}")
    print(f"  probe page:     {url}probe")
    print(f"  data root:      {config.DATA_ROOT}")
    print(f"  database:       {config.DB_PATH}")
    print(f"  log:            {config.LOGS_DIR / 'server.log'}")
    print("=" * 62)
    try:
        import qrcode

        q = qrcode.QRCode(border=1)
        q.add_data(url)
        q.make(fit=True)
        buf = io.StringIO()
        q.print_ascii(out=buf, invert=True)
        print(buf.getvalue())
    except Exception as exc:  # noqa: BLE001
        print(f"  (QR code unavailable: {exc})")
    print("  Laptop must be awake and on the same Wi-Fi. If it is asleep when")
    print("  you shoot, the photos wait on the phone and upload later - nothing")
    print("  is lost by uploading hours after the session.")
    print("=" * 62)
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-banner", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    setup_logging(a.verbose)
    config.ensure_dirs()
    con()
    if not a.no_banner:
        print_banner(a.host, a.port)
    log.info("starting on %s:%d  data_root=%s", a.host, a.port, config.DATA_ROOT)

    import uvicorn

    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
