"""Hand-mark control points - the held-out ground truth for register.py.

Click 8-10 permanent skin features per image (moles, freckles, small scars).
The SAME physical mole must carry the SAME id in every image, which is what
makes cross-image residuals meaningful - so the tool tracks a roster of ids
across the whole session and tells you which one to place next.

This is deliberately a phase 0 throwaway. OpenCV highgui rather than tkinter:
the OpenCV build here reports WIN32UI, it handles zoom/pan on a 4000 px image
without extra work, and it is already a dependency.

CONTROLS
  left click            place the pending id  (or grab a nearby point)
  left drag             nudge the grabbed point
  wheel                 zoom about the cursor
  right drag / arrows   pan
  [  ]                  previous / next pending id
  z                     undo last placement
  d                     delete the point nearest the cursor
  r                     reset the view
  s                     save
  q / esc               save and quit

    python -m src.controlpoints data/raw/calib/left60_main_01.jpg
    python -m src.controlpoints data/raw/calib          (walks the directory)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from . import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
WIN = "control points"
GRAB_PX = 14.0


def roster_ids() -> list[int]:
    """Every id used so far, across all marked images."""
    ids: set[int] = set()
    if config.CONTROLPOINTS_DIR.exists():
        for p in config.CONTROLPOINTS_DIR.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                ids.update(int(q["id"]) for q in d.get("points", []))
            except Exception:  # noqa: BLE001
                continue
    return sorted(ids)


class Marker:
    def __init__(self, path: Path):
        self.path = path
        self.img = cv2.imread(str(path))
        if self.img is None:
            raise SystemExit(f"could not read {path}")
        self.h, self.w = self.img.shape[:2]

        self.pts: dict[int, list[float]] = {}
        self.order: list[int] = []
        self.out = config.CONTROLPOINTS_DIR / f"{path.stem}.json"
        if self.out.exists():
            d = json.loads(self.out.read_text(encoding="utf-8"))
            for q in d.get("points", []):
                self.pts[int(q["id"])] = [float(q["x_px"]), float(q["y_px"])]
            self.order = list(self.pts)

        self.roster = roster_ids()
        self.pending = self._next_pending()

        self.zoom = 1.0
        self.cx, self.cy = self.w / 2.0, self.h / 2.0
        self.view_w, self.view_h = 1280, 860
        self.drag_id: int | None = None
        self.pan_from: tuple[float, float] | None = None
        self.mouse = (0.0, 0.0)
        self.dirty = False

    # --- id bookkeeping ---------------------------------------------------
    def _next_pending(self) -> int:
        for i in self.roster:
            if i not in self.pts:
                return i
        # Roster exhausted (or empty): allocate the next unused id overall, so a
        # newly discovered mole never collides with one already used elsewhere.
        return max([*self.pts, *self.roster], default=-1) + 1

    def step_pending(self, d: int) -> None:
        cands = sorted(set(self.roster) | set(self.pts) | {self._next_pending()})
        if not cands:
            return
        if self.pending in cands:
            i = cands.index(self.pending)
        else:
            i = 0
        self.pending = cands[(i + d) % len(cands)]

    # --- view -------------------------------------------------------------
    def to_img(self, sx: float, sy: float) -> tuple[float, float]:
        return (self.cx + (sx - self.view_w / 2) / self.zoom,
                self.cy + (sy - self.view_h / 2) / self.zoom)

    def to_screen(self, ix: float, iy: float) -> tuple[float, float]:
        return ((ix - self.cx) * self.zoom + self.view_w / 2,
                (iy - self.cy) * self.zoom + self.view_h / 2)

    def render(self) -> np.ndarray:
        M = np.array([[self.zoom, 0, self.view_w / 2 - self.cx * self.zoom],
                      [0, self.zoom, self.view_h / 2 - self.cy * self.zoom]], np.float32)
        canvas = cv2.warpAffine(
            self.img, M, (self.view_w, self.view_h),
            flags=cv2.INTER_NEAREST if self.zoom > 2 else cv2.INTER_AREA,
            borderValue=(30, 30, 30),
        )
        for pid, (x, y) in self.pts.items():
            sx, sy = self.to_screen(x, y)
            if not (-30 < sx < self.view_w + 30 and -30 < sy < self.view_h + 30):
                continue
            col = (0, 255, 255) if pid == self.pending else (0, 230, 0)
            cv2.circle(canvas, (int(sx), int(sy)), 11, col, 2, cv2.LINE_AA)
            cv2.drawMarker(canvas, (int(sx), int(sy)), col, cv2.MARKER_CROSS, 9, 1)
            cv2.putText(canvas, str(pid), (int(sx) + 13, int(sy) - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, str(pid), (int(sx) + 13, int(sy) - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

        # crosshair at the cursor, so placement is deliberate at high zoom
        mx, my = self.to_screen(*self.mouse)
        cv2.line(canvas, (int(mx) - 14, int(my)), (int(mx) + 14, int(my)), (200, 200, 200), 1)
        cv2.line(canvas, (int(mx), int(my) - 14), (int(mx), int(my) + 14), (200, 200, 200), 1)

        self._hud(canvas)
        return canvas

    def _hud(self, canvas: np.ndarray) -> None:
        placed = sorted(self.pts)
        missing = [i for i in self.roster if i not in self.pts]
        bar = np.zeros((96, self.view_w, 3), np.uint8)
        lines = [
            f"{self.path.name}   {self.w}x{self.h}   zoom {self.zoom:.2f}x   "
            f"{len(self.pts)} placed{'  *unsaved*' if self.dirty else ''}",
            f"PENDING ID: {self.pending}    placed: {placed}",
            (f"MISSING from this image: {missing}" if missing
             else "all roster ids placed" if self.roster else "first image - ids start at 0"),
        ]
        for i, t in enumerate(lines):
            col = (0, 255, 255) if i == 1 else ((120, 170, 255) if i == 2 and missing else (220, 220, 220))
            cv2.putText(bar, t, (10, 26 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        canvas[canvas.shape[0] - 96:] = cv2.addWeighted(
            canvas[canvas.shape[0] - 96:], 0.15, bar, 0.85, 0
        )

    # --- interaction ------------------------------------------------------
    def nearest(self, ix: float, iy: float) -> int | None:
        if not self.pts:
            return None
        pid = min(self.pts, key=lambda k: (self.pts[k][0] - ix) ** 2 + (self.pts[k][1] - iy) ** 2)
        x, y = self.pts[pid]
        return pid if np.hypot(x - ix, y - iy) * self.zoom <= GRAB_PX else None

    def on_mouse(self, ev, x, y, flags, _):
        ix, iy = self.to_img(x, y)
        self.mouse = (ix, iy)
        if ev == cv2.EVENT_LBUTTONDOWN:
            hit = self.nearest(ix, iy)
            if hit is not None:
                self.drag_id = hit
                self.pending = hit
            else:
                self.pts[self.pending] = [ix, iy]
                self.order.append(self.pending)
                self.dirty = True
                self.pending = self._next_pending()
        elif ev == cv2.EVENT_MOUSEMOVE:
            if self.drag_id is not None:
                self.pts[self.drag_id] = [ix, iy]
                self.dirty = True
            elif self.pan_from is not None:
                px, py = self.pan_from
                self.cx -= (ix - px)
                self.cy -= (iy - py)
        elif ev == cv2.EVENT_LBUTTONUP:
            self.drag_id = None
        elif ev == cv2.EVENT_RBUTTONDOWN:
            self.pan_from = (ix, iy)
        elif ev == cv2.EVENT_RBUTTONUP:
            self.pan_from = None
        elif ev == cv2.EVENT_MOUSEWHEEL:
            f = 1.25 if flags > 0 else 1 / 1.25
            new = float(np.clip(self.zoom * f, 0.05, 40.0))
            # keep the pixel under the cursor put
            self.cx += (ix - self.cx) * (1 - self.zoom / new)
            self.cy += (iy - self.cy) * (1 - self.zoom / new)
            self.zoom = new

    def save(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(
            json.dumps(
                {
                    "image": self.path.name,
                    "image_w": self.w,
                    "image_h": self.h,
                    "points": [
                        {"id": int(i), "x_px": round(float(p[0]), 2), "y_px": round(float(p[1]), 2)}
                        for i, p in sorted(self.pts.items())
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.dirty = False
        print(f"saved {len(self.pts)} points -> {self.out}")

    def fit(self) -> None:
        self.zoom = min(self.view_w / self.w, self.view_h / self.h)
        self.cx, self.cy = self.w / 2.0, self.h / 2.0


def mark(path: Path) -> None:
    m = Marker(path)
    m.fit()
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, m.on_mouse)
    print(f"\n{path.name}: roster {m.roster or '(empty - first image)'}  pending {m.pending}")
    while True:
        cv2.imshow(WIN, m.render())
        k = cv2.waitKey(16) & 0xFF
        if k in (ord("q"), 27):
            if m.dirty:
                m.save()
            break
        elif k == ord("s"):
            m.save()
        elif k == ord("z"):
            while m.order:
                last = m.order.pop()
                if last in m.pts:
                    del m.pts[last]
                    m.pending = last
                    m.dirty = True
                    break
        elif k == ord("d"):
            hit = m.nearest(*m.mouse)
            if hit is not None:
                del m.pts[hit]
                m.pending = hit
                m.dirty = True
        elif k == ord("["):
            m.step_pending(-1)
        elif k == ord("]"):
            m.step_pending(+1)
        elif k == ord("r"):
            m.fit()
        elif k in (ord("+"), ord("=")):
            m.zoom = min(40.0, m.zoom * 1.25)
        elif k in (ord("-"), ord("_")):
            m.zoom = max(0.05, m.zoom / 1.25)
        elif k == 81:
            m.cx -= 60 / m.zoom
        elif k == 83:
            m.cx += 60 / m.zoom
        elif k == 82:
            m.cy -= 60 / m.zoom
        elif k == 84:
            m.cy += 60 / m.zoom
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            if m.dirty:
                m.save()
            break
    cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="hand-mark control points")
    ap.add_argument("path", type=Path, nargs="?", default=config.CALIB_DIR)
    a = ap.parse_args(argv)
    config.ensure_dirs()

    if a.path.is_dir():
        paths = sorted(p for p in a.path.iterdir() if p.suffix.lower() in IMG_EXTS)
    else:
        paths = [a.path]
    if not paths:
        print(f"no images at {a.path}")
        return 2

    print(__doc__.split("CONTROLS")[1].split("    python")[0])
    for p in paths:
        mark(p)
    print("\nroster now:", roster_ids())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
