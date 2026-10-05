# Lesion Atlas: what we built and how it works

Snapshot: October 4, 2026

Lesion Atlas now combines a deployed skincare chatbot with a private personal
dataset workspace and an executable local machine-learning baseline. The
project's focus shifted from a single-person computer-vision research tool
toward an AI-first application with a path to user-specific models.

The chatbot works today. The personal ML pipeline can train, evaluate and make
experimental predictions, but it is not yet an accurate, complete acne-tracking
product. Calibration is not training, and training completion is not proof that
a model works for someone.

## What is available

| Component | Current state |
|---|---|
| Skincare and project chatbot | Deployed on Cloudflare; real replies tested |
| Chat integrated into local home page | Implemented; local relay successfully tested |
| Public personal model workspace | Deployed; final interactive browser verification pending |
| Local profiles, photo storage and annotation | Implemented in browser IndexedDB |
| Personal scale calibration | Manual marker-edge measurement |
| Personal CNN training and evaluation | Runnable locally in Python; tested with synthetic images |
| Experimental per-photo model predictions | Available through the Python command line |
| Today/Yesterday spot-count cards | Placeholders, not actual detection results |
| Automatic browser detection and cross-day tracking | Not implemented |
| Full legacy Python capture server hosted online | Not deployed; stays on the laptop |
| Custom domain | Not connected yet |

Public chatbot: https://lesion-atlas-chat.noahlee0318.workers.dev

Personal workspace: https://lesion-atlas-chat.noahlee0318.workers.dev/personal.html

Local homepage, while its server is running: http://127.0.0.1:8012/

Repository: https://github.com/noahlee0318/Lesion-Identifier

## The two different AI systems

The chatbot and the personal acne model are separate systems. They do not share
photos or automatically exchange a user's data.

**The chatbot** uses an existing language model hosted by Cloudflare. We built
the interface, backend integration, instructions, project context, limits, and
privacy boundaries. We did not train that language model.

**The personal ML baseline** is a small convolutional neural network trained
locally from a person's labeled photos. Its code performs actual optimization,
evaluation and inference. We have not supplied a pretrained acne detector or
established useful accuracy on real skin.

```text
Typed chat question
  → browser chat interface
  → Cloudflare Worker (through a local relay when using localhost)
  → Workers AI language model
  → plain-text answer

Personal photos
  → browser-local profile and IndexedDB storage
  → manual scale measurement and spot labels
  → private dataset export saved on the device
  → local Python CNN training
  → separate held-out evaluation
  → experimental per-photo predictions
```

## Chatbot: from a question to an answer

The chat interface is plain HTML, CSS and JavaScript. On the laptop it is
embedded directly in the home page. The public Cloudflare page uses the same
chat script. Visitors do not need a password or access code.

When someone sends a question, the browser includes that question and up to
four previous question/answer turns. It sends text only. There is no photo
attachment control and no automatic access to the journal, photo files,
annotations, counts, or local database.

On the public site, the request goes directly to `/api/chat` on the Cloudflare
Worker. On localhost, FastAPI receives the request and forwards it to the
configured HTTPS Worker. `CHAT_SERVICE_URL` tells the local server where that
Worker lives; the Windows user setting was configured during deployment.

The Worker checks the request's origin, size, message structure, and rate limit.
It rejects extra payload fields, client-supplied system messages, and supported
image-payload patterns. It then adds its own server-controlled instructions and
calls Workers AI through an AI binding. The default model is
`@cf/meta/llama-3.1-8b-instruct-fp8`. The browser never receives a Cloudflare
account API token.

The instructions keep the assistant focused on Lesion Atlas, photo setup,
acne, skincare ingredients, and routines. They tell it not to invent spot
counts, diagnose skin conditions, prescribe treatment, or claim it can inspect
photos. Short curated notes from the American Academy of Dermatology and NHS
provide grounding. This is not live web search, retrieval over a large document
database, or a guarantee that every answer is medically correct. Reference
links are background reading, not verified citations for every generated claim.

The response is rendered as plain text, never interpreted as HTML. Conversation
history stays in page memory and clears on reload; the application does not
save chat transcripts. Cloudflare still processes the submitted text under its
own service policies.

### Usage and error handling

The implementation limits individual messages to 2,000 characters, the request
conversation to 10,000 characters, and model output to 450 tokens. The configured
Cloudflare rate-limit binding permits eight requests per minute per Cloudflare
location for a shared application key. That is approximate distributed
throttling, not a hard global daily spending cap.

The project was set up for free-tier use, with no paid fallback. Public access
without a password was an explicit product choice; visitors can use the shared
allowance. Exhaustion or provider failures produce plain-English errors.

We fixed a concrete local-connection problem: Cloudflare rejected Python's
default urllib client identification with HTTP 403 / error 1010. The relay now
identifies itself as `LesionAtlas/1.0`. A real reply subsequently succeeded
through the local endpoint used by the home-page chat.

## Personal workspace: preparing a private dataset

Each person creates a local profile, then adds JPEG or PNG photos with a date,
view, and purpose. Views are front, left 60 degrees, or right 60 degrees; side
views remain experimental. Practice calibration photos are separate from
session photos and are excluded from ML training.

The browser stores original image bytes in IndexedDB. Display previews do not
replace or re-encode the stored originals. Profiles separate records for
organization, but are not authenticated accounts: another person using the same
browser can access that browser's profiles.

The workspace has no photo-upload endpoint. Its Content Security Policy permits
only same-origin connections for the integrated text-only chatbot. Photo bytes, labels, profile names and calibration
data are not sent to the chatbot or Cloudflare by this workspace.

### Calibration

A user enters the measured length of a printed marker edge in millimetres and
taps its two endpoints in the photo. The app computes:

```text
pixels per millimetre = distance between tapped endpoints / measured length
```

For example, a 30 mm edge spanning 780 image pixels gives 26 pixels per mm.
That scale converts the chosen spot diameter into an annotation radius and
later converts predicted coordinates into millimetres.

This is manual scale measurement. It does not automatically verify pose,
lighting, perspective, registration quality, or whether a model understands
that person's acne. A face is curved, so a single scale is an approximation.
The marker should be near the cheek's distance from the camera.

### Labeling and review

After measuring scale, the user chooses a spot diameter in millimetres and taps
each spot center. Labels are saved in oriented full-image coordinates, not
display-preview coordinates. A photo only becomes eligible for training after
the user explicitly marks it reviewed.

A reviewed photo with no labels means that no acne spots were visible. An
unreviewed photo means the labels are not ready; the trainer must not interpret
it as a negative example. Users should leave uncertain photos unreviewed and
save drafts before switching photos or profiles.

### Export and restore

A profile export is one JSON file containing original bytes as base64,
SHA-256 image hashes, profile identity, dates, views, scale, labels and review
states. It is private data and belongs outside Git and cloud-synced folders.

Restore checks byte hashes, dimensions and annotation structure, then creates
a separate local profile. It does not overwrite an existing profile. Restored
profiles get new identifiers; retain the original frozen export to reproduce
an old training run.

Current limits are 20 MB per photo, 200 MB of original photos per profile,
and 300 MB per imported backup. Large JSON files need substantial browser
memory. Browser data can be lost if storage is cleared or evicted, so keep
original photos and local exports.

Storage belongs to a particular device, browser and origin. The public domain
and localhost have different datasets. A domain change or another device needs
an explicit export/restore. Use HTTPS or localhost; plain HTTP LAN access does
not support the required browser cryptography tools.

## Local ML: how personalization actually happens

The Python pipeline uses one person's reviewed session photos from one view
at a time. It rejects mixed-profile exports, duplicate original image bytes,
invalid scales, invalid coordinates, and image/hash mismatches. It does not
silently combine another user's photos with the current person's data.

### Date-separated training, validation and test sets

Splitting by random photo would risk placing almost identical photos in both
training and testing. Instead, the new pipeline uses chronological calendar
windows:

```text
older training days | 3-day gap | 14 validation days | 3-day gap | 14 test days
```

Gap days are excluded. Roughly 35 calendar days are needed, with reviewed
photos in each block and positive examples in training and validation. Sparse
capture may require longer. Readiness means a training experiment can start;
it does not guarantee sufficient data or accuracy.

### Training

The experimental baseline resizes an in-memory RGB preview to 256×256 and
turns reviewed annotation discs into a binary target mask. A three-layer
convolutional network learns to assign lesion probabilities to pixels.

It uses Adam, weighted binary cross-entropy with positive weight 10, and a
recorded seed of 17. It trains from scratch: there are no downloaded pretrained
weights. CUDA is the default, and the command errors if it is unavailable;
CPU training must be selected explicitly.

After training, thresholds of 0.3, 0.5 and 0.7 are compared on validation days.
The highest validation F1 selects the threshold. Test days do not choose it.

This small whole-image baseline is distinct from the planned tiled production
detector. Resizing can erase small lesions, and adjacent lesions may merge.
It provides a real executable ML foundation, not a claim of useful accuracy.

### Evaluation and predictions

Thresholded mask components containing at least two preview pixels become
candidate centers. The evaluator matches predicted centers one-to-one with
labels, within 2 mm, and pools true positives, false positives and false
negatives across test photos. Negative photos contribute false positives too.

Precision answers how many predictions matched labeled spots. Recall answers
how many labeled spots were found. Undefined values are reported as `null`,
not as perfect performance.

The provisional engineering gate requires precision above 0.8 and recall above
0.7, at least 20 labeled test lesions, and at least five test capture days.
Outcomes are `provisional_gate_passed`, `gate_failed`, or
`insufficient_evidence`. These sample minimums do not prove statistical power
or clinical validity. Repeatedly choosing models based on the same test set
would bias the result; later comparisons need untouched future days.

The prediction command returns candidate centers in millimetres relative to
one photo. Those are not registered anatomical coordinates, confirmed acne
diagnoses, or trusted daily totals. Summing different views would double-count
some face regions. No such aggregation is connected to the count cards.

### Reproducible run artifacts

| File | Purpose |
|---|---|
| `weights.pt` | Learned model parameters |
| `model.json` | Architecture, profile/view, hashes, split IDs, day counts, training settings, losses, selected threshold and validation metrics |
| `evaluation.json` | Test metrics, per-image results, sample counts and provisional gate verdict |

A new run refuses to overwrite an existing output directory and must live
outside the code repository. Evaluation requires the exact dataset snapshot
used for training, checks the split metadata and weights, and refuses to
overwrite its report. A recorded seed is not a promise of bit-exact results
across GPU hardware and runtime versions.

## Verification and its limits

The latest implementation passed 58 automated checks: 10 Python unittest
checks, 21 legacy tiling checks, 19 legacy split checks, and eight Node checks.
They cover request boundaries, date separation, profile isolation, changed
datasets/weights, immutable output handling, and actual synthetic optimization,
evaluation and prediction.

A separate default-resolution CPU pass used 40 generated images, including
empty photos and one/two-spot cases. Two epochs reduced loss from 0.8228 to
0.7972, but test detection remained poor: 0 true positives, 14 false positives,
and 14 false negatives. The report returned `insufficient_evidence` because
there were only 14 labeled test lesions. That is software-path verification,
not a real-skin result or evidence of detector quality.

Four generic live chatbot requests also succeeded: project/count availability,
skincare, an unrelated coding request, and a diagnosis request. The assistant
acknowledged unavailable counts and declined coding and diagnosis. These smoke
checks do not establish comprehensive medical accuracy or robust resistance
to all prompt attacks.

The final personal-workspace browser check was blocked by automatic approval
review because of an earlier browser usage-limit failure. Interactive visual,
annotation and restore checks remain pending. Do not confuse passing unit
tests or successful deployment with a fully verified browser workflow.

## Deployment and code organization

Cloudflare hosts the Worker and the code assets under `cloudflare/public`.
Photos and personal model weights are not deployment assets. The Python
capture server, SQLite database, local training and legacy research tools
remain on the laptop. The public site does not run that Python backend.

| Location | Responsibility |
|---|---|
| `cloudflare/worker.mjs` | Chat request validation, limits and Workers AI calls |
| `cloudflare/context.mjs` | Chat scope, project facts and reference notes |
| `cloudflare/public/chat.js` | Shared chat interactions and memory-only history |
| `cloudflare/public/personal*` | Local dataset workspace, scale/labels and readiness |
| `src/server/chat.py` | Text-only local relay to Cloudflare |
| `src/server/static/index.html` | Local homepage, chat and workspace link |
| `src/personal_ml.py` | Dataset validation, CNN training, evaluation and prediction |
| `tests/` | Automated boundary, geometry, split and ML checks |

The chatbot, integration fix, password removal, connection fix and personal ML
structure were pushed to GitHub `main`. The personal-workspace implementation
was committed as `eb0cfbb` and deployed to the existing Cloudflare service.
Documentation updates may follow that implementation commit.

## Existing research tools versus new work

The repository already contained marker generation, landmark checks, registration
experiments, photo QA/ingest, tiling and date-split foundations. Those should not
be described as a newly completed production acne detector.

Earlier review work on registration, tiling and splits exists on the separate
`review-fixes-register-tiling-splits` branch. It was not merged into `main` as
part of this chatbot/personalization work. The new personal ML module implements
its own fixed-window split; that does not imply the legacy split module has
received every review fix.

## Next work, in practical order

1. Interactively verify the workspace on desktop and mobile with disposable
   photos, including export/restore and orientation handling.
2. Improve annotation usability and establish a consistent labeling rubric.
3. Gather and review real personal data; measure label agreement and model
   errors rather than assuming calibration makes the model accurate.
4. Replace or improve the baseline with a detector that retains fine detail
   and uses personal examples efficiently.
5. Add a browser-compatible inference runtime and an explicit model-review
   workflow before enabling automatic counts.
6. Implement geometric registration, region ownership and cross-day identity
   tracking before claiming longitudinal lesion tracking.
7. Attach the public interface to a chosen subdomain when ready. Preserve local
   photo storage and plan export/restore when changing origins.

The accurate project description today is: **a deployed AI skincare assistant,
a private browser-local annotation workspace, and a reproducible experimental
pipeline for training and evaluating personal lesion models.** It is not yet
an accurate autonomous acne detector, a diagnostic tool, or a complete tracker.

Detailed guides: [AI integration](AI_CHATBOT.md) and
[personal ML commands and workflow](PERSONAL_ML.md).
