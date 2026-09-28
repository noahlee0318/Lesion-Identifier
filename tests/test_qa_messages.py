"""The rejection messages are a user interface, so they get tested like one.

A photo is rejected while the rig is still set up or it is not reshot at all,
and a message nobody understands gets shrugged at and overridden. That is how
a bad frame enters a longitudinal series - which is the single thing the QA
pass exists to prevent. So the wording is a correctness property here, not
polish, and this file pins it.

Runs standalone (no pytest needed):
    python -m tests.test_qa_messages
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.server import qa  # noqa: E402


# Words that mean nothing to someone holding a phone at 7am. If one of these
# has to appear, it belongs in `detail`, never in `what` or `fix`.
JARGON = [
    "laplacian", "variance", "var ", "clipped", "clipping", "aruco", "fiducial",
    "px/mm", "luma", "chroma", "histogram", "kernel", "threshold", "decode",
    "codec", "buffer", "null", "none", "exception", "traceback", "http",
    "json", "sha256", "stderr", "opencv", "cv2", "numpy", "float", "int(",
    "blown", "crushed", "highlights cl", "quantile", "percentile", "sigma",
]


def _jpeg(img: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def _flat(value: int = 128, size=(900, 700)) -> np.ndarray:
    return np.full((size[0], size[1], 3), value, np.uint8)


def _all_problems() -> list[qa.Problem]:
    """One of every problem the module can emit, gathered from real checks."""
    out: list[qa.Problem] = []

    # not an image at all
    out += qa.check(b"this is not a jpeg at all").problems or []
    # flat grey: blurry, and no marker
    out += qa.check(_jpeg(_flat(128))).problems or []
    # pure white: blurry, blown, no marker
    out += qa.check(_jpeg(_flat(255))).problems or []
    # pure black: blurry, dark, no marker
    out += qa.check(_jpeg(_flat(0))).problems or []

    assert len(out) >= 4, "the fixtures did not trigger enough distinct problems"
    return out


# --------------------------------------------------------------------------
# the contract
# --------------------------------------------------------------------------

def test_every_problem_has_a_what_and_a_fix():
    for p in _all_problems():
        assert p.what.strip(), f"problem with no plain-English `what`: {p}"
        assert p.fix.strip(), (
            f"problem with no `fix`: {p.what!r} - a rejection the reader cannot "
            "act on is the same as no rejection"
        )


def test_no_jargon_in_what_or_fix():
    for p in _all_problems():
        for field, text in (("what", p.what), ("fix", p.fix)):
            low = text.lower()
            for bad in JARGON:
                assert bad not in low, (
                    f"{field} contains {bad!r}, which is jargon:\n    {text}"
                )


def test_the_numbers_survive_in_detail():
    """Plain English is about ordering, not about hiding the evidence.

    The thresholds are provisional until real photos exist. When one turns out
    wrong, the measurement is the only way to find out.
    """
    problems = _all_problems()
    assert all(p.detail for p in problems), "a problem carried no detail at all"
    # The decode failure has no measurement to report - there was no image to
    # measure. Every problem that came from a threshold comparison must show
    # the number it compared.
    numeric = [p for p in problems if re.search(r"\d", p.detail)]
    assert len(numeric) >= 3, (
        f"only {len(numeric)} problems carried a number; the thresholds are "
        "provisional and the measurement is how a wrong one gets caught"
    )


def test_what_reads_as_a_sentence():
    for p in _all_problems():
        assert p.what[0].isupper(), f"`what` does not start a sentence: {p.what!r}"
        assert p.what.rstrip().endswith("."), f"`what` is not a sentence: {p.what!r}"


def test_fix_tells_the_reader_to_do_something():
    """Every fix names a physical action, not a state of the world."""
    verbs = ("tap", "move", "step", "switch", "bring", "add", "get", "take",
             "check", "hold", "put", "tape", "shoot", "set", "point", "back")
    for p in _all_problems():
        low = p.fix.lower()
        assert any(v in low for v in verbs), (
            f"fix names no action the reader can take: {p.fix!r}"
        )


def test_reasons_are_the_plain_english_sentences():
    """Anything still reading `reasons` - the server log - gets plain English too."""
    r = qa.check(_jpeg(_flat(255)))
    assert r.reasons and r.problems
    assert len(r.reasons) == len(r.problems)
    for sentence, p in zip(r.reasons, r.problems):
        assert sentence == p.as_sentence()
        low = sentence.lower()
        for bad in JARGON:
            assert bad not in low, f"log line contains jargon {bad!r}: {sentence}"


def test_problems_survive_the_json_round_trip():
    """The UI reads this dict; a missing key renders an empty rejection."""
    d = qa.check(_jpeg(_flat(255))).to_dict()
    assert d["problems"], "problems vanished from the wire format"
    for p in d["problems"]:
        assert set(p) == {"what", "fix", "detail"}
        assert isinstance(p["what"], str) and isinstance(p["fix"], str)
    assert d["reasons"], "reasons must stay populated for older readers"


# --------------------------------------------------------------------------
# the checks still detect the right things
# --------------------------------------------------------------------------

def test_a_blank_frame_is_rejected_as_out_of_focus():
    r = qa.check(_jpeg(_flat(128)))
    assert r.verdict == "FAIL"
    whats = " ".join(p.what.lower() for p in r.problems)
    assert "focus" in whats


def test_a_white_frame_is_rejected_as_too_bright():
    r = qa.check(_jpeg(_flat(255)))
    whats = " ".join(p.what.lower() for p in r.problems)
    assert "too bright" in whats, whats


def test_a_black_frame_is_rejected_as_too_dark():
    r = qa.check(_jpeg(_flat(0)))
    whats = " ".join(p.what.lower() for p in r.problems)
    assert "too dark" in whats, whats


def test_a_missing_marker_says_marker_not_millimetres():
    r = qa.check(_jpeg(_flat(128)))
    marker = [p for p in r.problems if "marker" in p.what.lower()]
    assert marker, "no problem mentioned the printed marker"
    assert "millimetre" in marker[0].fix.lower(), (
        "the fix should say why the marker matters, in units a person reads"
    )


def test_unreadable_bytes_are_rejected_without_crashing():
    r = qa.check(b"\x00\x01\x02 not an image")
    assert r.verdict == "FAIL"
    assert len(r.problems) == 1
    assert "photo" in r.problems[0].what.lower()
    assert "most compatible" in r.problems[0].fix.lower(), (
        "the HEIC-vs-JPEG setting is the actual cause here and should be named"
    )


def test_require_fiducial_false_skips_the_marker_problem():
    r = qa.check(_jpeg(_flat(128)), require_fiducial=False)
    assert not any("marker" in p.what.lower() for p in r.problems)


def test_a_clean_frame_with_a_marker_passes():
    """A real marker on a textured field: nothing to complain about."""
    rng = np.random.default_rng(4)
    img = rng.integers(60, 200, (1400, 1100, 3), dtype=np.uint8)  # texture = sharp

    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT_NAME))
    side = 420                                   # big enough to clear MIN_LESION_PX
    m = cv2.aruco.generateImageMarker(adict, config.ARUCO_ID, side)
    # Matte paper under a lamp is not 255, and ink is not 0.
    m = np.where(m > 127, 238, 28).astype(np.uint8)
    m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    quiet = 60
    img[40:40 + side + 2 * quiet, 40:40 + side + 2 * quiet] = 238
    img[40 + quiet:40 + quiet + side, 40 + quiet:40 + quiet + side] = m

    r = qa.check(_jpeg(img))
    assert r.aruco_found, "the fixture marker was not detected - fixture problem"
    assert r.verdict == "PASS", [p.what for p in r.problems]
    assert r.problems == []


# --------------------------------------------------------------------------
# the exposure check measures skin, not the marker
# --------------------------------------------------------------------------

def _frame_with_marker(side: int = 780, paper: int = 255, ink: int = 0,
                       shape=(3024, 4032)) -> np.ndarray:
    """A textured frame carrying a marker at a realistic relative size.

    780 px is a 30 mm marker at 26 px/mm, which is the geometry SETUP_NUMBERS
    targets; the quiet zone is the printed 8 mm.
    """
    rng = np.random.default_rng(11)
    img = rng.integers(70, 190, (shape[0], shape[1], 3), dtype=np.uint8)
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICT_NAME))
    m = cv2.aruco.generateImageMarker(adict, config.ARUCO_ID, side)
    m = cv2.cvtColor(np.where(m > 127, paper, ink).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    quiet = int(round(side * 8 / 30))
    y0 = x0 = 100
    img[y0:y0 + side + 2 * quiet, x0:x0 + side + 2 * quiet] = paper
    img[y0 + quiet:y0 + quiet + side, x0 + quiet:x0 + quiet + side] = m
    return img


def test_the_marker_does_not_trip_the_exposure_check():
    """It is white paper and black ink, in every frame, on purpose.

    At the rig's geometry the marker plus its quiet zone is ~12% of the frame
    and its white alone is ~6.5% - over triple the 2% limit. Measuring it
    would reject essentially every correctly exposed photo and blame the lamp,
    which trains the reader to override the verdict.
    """
    img = _frame_with_marker(paper=255, ink=0)          # worst case: fully clipped
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raw_high = float((gray >= qa.HI).mean())
    assert raw_high > config.QA_MAX_CLIPPED_HIGH * 2, (
        f"fixture does not reproduce the problem: {raw_high:.1%} pure white"
    )

    r = qa.check(_jpeg(img))
    assert r.aruco_found, "fixture marker not detected"
    assert r.clipped_high_frac < config.QA_MAX_CLIPPED_HIGH, (
        f"the marker itself was counted as blown skin: {r.clipped_high_frac:.1%}"
    )
    assert not any("too bright" in p.what.lower() for p in r.problems), (
        [p.what for p in r.problems]
    )


def test_genuinely_blown_skin_is_still_caught_with_a_marker_present():
    """Masking the marker must not mask the thing the check is for."""
    img = _frame_with_marker(paper=238, ink=28)
    img[1500:2600, 1500:3600] = 255                     # a big blown patch of face
    r = qa.check(_jpeg(img))
    assert r.aruco_found
    assert any("too bright" in p.what.lower() for p in r.problems), (
        f"blown skin went unreported ({r.clipped_high_frac:.1%} over limit "
        f"{config.QA_MAX_CLIPPED_HIGH:.1%})"
    )


def test_exposure_falls_back_to_the_whole_frame_with_no_marker():
    r = qa.check(_jpeg(_flat(255)), require_fiducial=False)
    assert not r.aruco_found
    assert r.clipped_high_frac == 1.0, (
        "with no marker there is nothing to exclude and the whole frame counts"
    )


# --------------------------------------------------------------------------
# the page renders what the server sends
# --------------------------------------------------------------------------

def test_the_page_renders_problems_not_raw_reason_strings():
    page = (Path(__file__).resolve().parent.parent
            / "src" / "server" / "static" / "index.html").read_text(encoding="utf-8")
    assert "problemsHtml" in page, "the page does not render the structured problems"
    assert "q.problems" in page
    assert "Do this:" in page, "the page does not surface the fix"
    assert "esc(" in page, "problem text is interpolated without escaping"
    # The measurements stay reachable, just not first.
    assert "<summary>measurements</summary>" in page


def test_the_page_no_longer_shouts_fail():
    page = (Path(__file__).resolve().parent.parent
            / "src" / "server" / "static" / "index.html").read_text(encoding="utf-8")
    assert "RESHOOT" in page, "the verdict word should say what to do"


# --------------------------------------------------------------------------

def _run() -> int:
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")
    print("-" * 62)
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    print("qa messages")
    print("-" * 62)
    raise SystemExit(_run())
