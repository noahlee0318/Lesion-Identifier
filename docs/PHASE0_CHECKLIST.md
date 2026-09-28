# Phase 0 — physical checklist

Everything on this page is yours. None of it can be done in software, and
until it is done there is nothing for the phase 0 scripts to measure.

Work top to bottom. Steps 1–4 are one sitting (~45 min). Step 5 is spread
over three days and is the only part you cannot compress.

---

## 1 · Print the fiducial

```powershell
python -m src.fiducial generate
```

Writes `docs/fiducial/calibration_target.pdf` (and `.png`, and a sidecar JSON
recording every patch value).

- [ ] Print the **PDF**, on **matte** paper. Gloss speculars under the lamp
      and destroys the white-balance reference.
- [ ] In the print dialog: **100% / Actual Size**. Turn OFF "Fit to page",
      "Shrink to fit", "Scale to fit media". This is the single most common
      way to silently break every millimetre in the project.
- [ ] **Put a ruler on the printed marker.** Outer black edge must read
      **30.0 mm**. The reference line must read **100.0 mm**. If either is
      off, reprint with scaling disabled. Do not proceed on a bad print.
- [ ] Trim on the crop marks.

Then photograph the printed sheet, square on, and check it:

```powershell
python -m src.fiducial verify path\to\photo_of_sheet.jpg
```

Wants PASS. If the four edges disagree, you were not square-on to the paper —
re-shoot straight before concluding the print is wrong.

---

## 2 · Mount the fiducial

- [ ] Tape it to a **headband** (or a small stand that sits in frame).
- [ ] It must sit **flat**, and as close as you can manage to the **same
      focal plane as your cheek** — not angled away, not tucked behind your ear.
- [ ] **Check all three poses now, before you commit.** A marker on the left
      temple vanishes in the right-60 shot. If one mount cannot cover all
      three, use two markers or move it to the forehead centre.
- [ ] It must be **in focus** in every pose. An out-of-focus marker still
      detects but its corners blur, and px/mm is the divisor on every
      measurement you will ever report.

---

## 3 · Build the rig

- [ ] **Blinds shut. Overhead lights off.** Daylight is the confounder that
      will put a step change in your chart on the day the weather turns.
- [ ] **One diffused lamp**, one fixed position. Diffused, not bare — a bare
      bulb speculars off skin and blows out exactly the raised lesions you
      are trying to measure.
- [ ] **Floor tape** for your feet.
- [ ] **Wall marks left and right at 60°** to turn toward. Not "about 60" —
      mark them. See the geometry note below.
- [ ] Pick a **time of day** and keep it. Erythema tracks temperature,
      exercise and time since showering, all of which track the clock.
- [ ] iPhone: **Settings → Camera → Formats → Most Compatible** (writes JPEG,
      not HEIC) and **Settings → Photos → Keep Originals**.
- [ ] Record every number in `SETUP_NUMBERS.md` **as you go**. You will need
      to rebuild this exactly after something gets knocked over, and you will
      not remember.

### Marking 60° without a protractor

Stand at the tape. Measure the distance `d` from you to the wall you face.
The mark goes `d × tan(60°) = d × 1.732` to the side, at eye height.

At d = 1.5 m → mark at **2.60 m** to each side.

If the wall is too narrow, use two floor marks instead: stand on the tape,
turn until your shoulders square to a point `1.732 × d` off-axis.

---

## 4 · Verify the upload path — before any real session

- [ ] Start the server (or install autostart — see `NETWORK_SETUP.md`):
      ```powershell
      python -m src.server.main
      ```
- [ ] From the phone, browse to the URL it prints (or scan the QR code).
- [ ] Shoot one throwaway photo. Copy it to the laptop **over USB**, and
      upload **the same photo** through the page.
- [ ] Run the fidelity check:
      ```powershell
      python scripts\verify_upload_fidelity.py --usb IMG_XXXX.JPG --uploaded <stored path>
      ```
- [ ] **Read the verdict.** If it says Safari resized the file, stop and fix
      that before shooting anything real — a resize changes px/mm and
      silently corrupts every measurement downstream.

Also do the stream probe while you are there, with the printed sheet at ~30 cm:

- [ ] Open `/probe` on the phone, grab a frame, read the px/mm result.
      It decides whether phase 1 keeps the ghost overlay. Expect Safari to
      refuse it over plain HTTP — that refusal is itself the finding.

---

## 5 · Shoot the calibration set

**This is calibration, not a session.** One pose only.

> **5 repeats total, spread over at least 3 days. Not 5 per day.**
> Each repeat is a fresh setup — walk away, take the headband off, come back
> and rebuild the rig from `SETUP_NUMBERS.md`. Repeatability is the entire
> point; five shots in one sitting tell you nothing about whether the pose
> lands in the same place tomorrow.

For each of the 5 repeats:

- [ ] Rebuild the rig from scratch (tape, lamp, headband, wall marks)
- [ ] Shoot **left 60°** on the **main lens** at ~30 cm
- [ ] Shoot **left 60°** on the **2× lens** at ~55 cm — same framing, farther away
- [ ] Upload both through the page in **Phase 0 calibration** mode

That is **10 files**. The page shows a live counter — repeats done, days
covered, and which lens pairs are incomplete.

Suggested spread:

| Day | Repeats | Files |
|---|---|---|
| 1 | 2 | 4 |
| 2 | 2 | 4 |
| 3 | 1 | 2 |

- [ ] Tap to focus on the **cheek**, not the centre of the frame.
- [ ] Marker fully in frame and in focus, every single shot.
- [ ] If a pose comes back **FAIL**, reshoot it in the same sitting, while
      the lamp and the tape are still where they were.

---

## 6 · Hand off

When the counter reads 5/5 across ≥3 days, the page moves to **step 2** and
tells you to run `facemesh_check`. Tell me and I will run:

1. `facemesh_check.py` → **the angle verdict**: hold at 60° or drop to 50°.
   Then you tap the matching **Lock at __°** button, which opens daily capture.
2. `verify_upload_fidelity.py` + the `/probe` numbers → phase 1 architecture
3. then you hand-mark control points and I run the registration bake-off
   → **GO/NO-GO** against median < 1.5 mm, p95 < 3 mm, on residuals **pooled
   across every pair**

   **Mark 8–10 points per image, on every calibration frame.** This is the one
   step where your effort changes what the gate can say. The p95 half of the
   gate is refused below **20 pooled residuals** — below that the number is
   just the worst single point wearing a percentile's name, and the verdict
   comes back INCONCLUSIVE rather than GO. Five sittings at 10 points each is
   ~40 pooled residuals, which is comfortable. Five sittings at 3 points each
   is 12, which is not a gate.

   Pick things that are unambiguous on every frame — moles, a scar, a
   distinctive freckle. A point you have to guess at is worse than one fewer
   point, because it adds error the method never made.
4. `lens_compare.py` → main vs 2×

---

## The one ordering rule

**Do not start real 3-pose sessions until the angle is locked.**

If phase 0 pushes you from 60° to 50°, every session shot at 60° beforehand
is at the wrong angle and cannot join the series.

**So: do NOT start the daily 3-pose sessions during calibration.**

### When exactly does daily capture start?

The day after `facemesh_check` passes and you lock the angle. Not a fixed
calendar day — whenever that lands, which is ~3–4 days from rig-up.

**The upload page enforces this.** The Daily session tab stays greyed out
until you tap **Lock at 60°** (or 50°), which only appears once all 5 repeats
are in across 3+ days. The banner at the top of the page always says which of
the three steps you are on and what to do next.

It is a guardrail, not a wall — there is an "upload a session anyway"
override behind a confirmation, because a blocked upload of a real session
would be a permanently lost day. Use it only if the counter is wrong.

```
day 1   rig up · repeats 1–2 · both lenses
day 2   repeats 3–4
day 3   repeat 5  ->  facemesh_check runs  ->  ANGLE LOCKED
day 4   first real 3-pose session. Never stops after this.
```

> **Note on a contradiction in the source docs.** The runbook narrative says
> "days 1–2, shoot the five repeats… day 3 onward, start real sessions",
> while the build plan says the five repeats spread across **three days**.
> Both cannot be true. The resolving principle is the rule above: **the gate
> is the angle lock, not the calendar.** Follow the table.

### Why not start frontal-only shots now?

The 60°-vs-50° decision does not touch the frontal pose, so in principle you
could. Don't. The rig itself — lamp position, distance, fiducial mount — is
still moving during calibration, and a session shot before the rig is locked
carries different lighting and scale from everything after it. That is
exactly the lighting-drift confounder this project exists to eliminate, and
it is invisible until month three.

Three days is cheap. A step change in your chart is not.

### After the lock

Daily capture starts and never stops — day 1 of the longitudinal series and
the first of your training images. Shoot with the normal Camera app into a
3-pose session and upload through the page in **Daily session** mode; the
capture app in phase 1 is automation of something you can already do by hand.

Expect roughly 5–8 manually-shot sessions before the app takes over.

Calendar time is the one input you cannot buy back later. Sixty days of
history takes sixty days, and no effort in week 6 recovers a session you did
not shoot in week 1.
