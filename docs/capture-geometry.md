# Capture geometry — why 60°

The locked decision and the arithmetic behind it, so it does not get
re-derived every time someone looks at the repo.

---

## The projection problem

Skin at angle θ to the sensor projects at **cos(θ)** of its true size.

For a point on your face, the incidence angle is roughly

```
incidence ≈ |φ − yaw|
```

where **φ** is that point's azimuth from straight-ahead: nose at 0°, directly
lateral at 90°.

| Region | φ | Frontal | 45° yaw | **60° yaw** | 90° yaw |
|---|---|---|---|---|---|
| Forehead, centre | ~5° | 5° | 40° | 55° | 85° |
| Medial cheek | ~30° | 30° | 15° | 30° | 60° |
| Lateral cheek | ~60° | 60° | 15° | **0°** | 30° |
| Preauricular / sideburn | ~80° | 80° | 35° | **20°** | 10° |

Incidence in each column. Lower is better.

**A 3 mm papule at the sideburn projects to about 0.5 mm frontally, and
2.8 mm at 60° yaw.**

Frontally, a lesion by your sideburn occupies half a millimetre of projected
width, smeared over a handful of pixels and lit at grazing incidence, so it
reads as either a blown specular strip or a shadow. It is not recoverable.

---

## Zoom does not fix this

A tighter crop shows **the same angle, bigger**. Foreshortening is a property
of the geometry between your skin and the sensor, not of magnification.
Turning your head fixes it. Zoom does not.

Do not propose optical zoom as a fix for coverage. The lens question in phase
0 is about depth of field and working distance, which is a different problem.

---

## Why not 90° (full profile)

A three-quarter angle is not a compromise between frontal and profile. It
roughly **bisects the span of face you need on one side**, so everything from
mid-cheek to sideburn lands at workable incidence at once.

A full profile optimises the sideburn alone and costs three things:

1. **Landmarks.** MediaPipe Face Mesh holds to roughly ±60° yaw. Past that,
   half the landmarks self-occlude and you are back to landmark-free
   registration — the hardest version of this problem.

2. **The overlap band.** At 60°, the band around φ ≈ 30–50° (mid-cheek) is
   well-seen in *both* the frontal and the turned pose. That shared skin is
   what ties the three pose coordinate frames into one head, and it yields a
   registration-error estimate that reports itself **every single day**.
   Frontal and profile share almost nothing.

3. **The medial cheek**, which lands at 60° incidence in a profile — worse
   than the frontal already gives you.

The one genuine advantage of a profile is less depth variation, so depth of
field is easier. Not enough to outweigh the above.

---

## Why not 45°

60° drops the sideburn from 35° to 20° incidence and puts the lateral cheek
at 0°, while staying inside Face Mesh's working range. 45° gives up most of
the sideburn gain, which is the region that motivated the whole design.

---

## The one open geometry question

**If phase 0 shows Face Mesh does not fire reliably at 60°, step down to 50°.**

That constraint sets the angle — not aesthetics, not preference. This is what
`facemesh_check.py` measures, and it is the first gate in the project.

---

## Region ownership

Because the poses overlap, every region needs exactly one owner, or the
mid-cheek band gets counted twice.

| Pose | Owns — counted here | Also visible — validation only |
|---|---|---|
| Frontal | forehead · chin & perioral | medial cheeks, nose |
| Left 60° | left cheek (to jawline) · left temple & sideburn | mid-cheek band, left forehead edge |
| Right 60° | right cheek (to jawline) · right temple & sideburn | mid-cheek band, right forehead edge |

A detection in a pose that does not own that region is kept with
`owned = false`, used for the daily cross-pose check, and **excluded from
every count**. That one flag is what stops double-counting the overlap.

Temple and sideburn are split out from cheek rather than folded into it:
buried inside "cheek", the area you care about most could never be charted on
its own.

---

## Shooting notes

- **Shoot from farther on the 2× lens rather than close on the main lens** —
  same framing, more depth of field, much less perspective distortion. Depth
  of field is a real constraint in a turned pose: your nose and ear sit ~10 cm
  apart in depth and a phone at 30 cm cannot hold both sharp. But the tele
  sensor is often lower quality, so **measure it** rather than assuming.
  That is phase 0 question (c).

- **Fix the yaw physically.** Floor tape for your feet, plus a wall mark to
  each side to turn toward. "About 60°" drifting week to week reintroduces
  exactly the variance you are eliminating.

  At distance `d` to the facing wall, the 60° mark sits `d × tan(60°) =
  d × 1.732` to the side.

- **Tap to focus on the region the pose owns**, not the centre of the frame.

---

## If 60° still under-covers the sideburn

Add full 90° profiles as a **fourth and fifth pose owning only the temple and
sideburn regions** — do not replace the three-quarters.

Same for a chin-up shot if the underjaw turns out to matter.

Add either **only on evidence from real photos, never preemptively**. Each
extra pose is daily friction, and friction is the top risk in the project.
