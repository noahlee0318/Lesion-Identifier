# How the Lesion Atlas chatbot works

The assistant answers questions about using Lesion Atlas, taking tracking
photos, acne, skincare ingredients, and routines. It uses Cloudflare Workers AI.
It cannot examine photos, access the tracking database, or count lesions.

## Architecture

```text
Local upload page -> /chat/ -> local text-only relay -> Cloudflare Worker
                                                        |
Cloudflare-hosted chat page -----------------------------+
                                                        |
                                                Workers AI model
```

The same HTML/CSS/JavaScript chat page runs locally and on Cloudflare. Local
FastAPI serves it at `/chat/`; its `/api/chat` route forwards validated text to
the HTTPS Worker configured in `CHAT_SERVICE_URL`. On Cloudflare the browser
calls the Worker directly on the same origin. Only `cloudflare/public` is the
static deployment directory. The Python photo-ingestion server stays local.

`cloudflare/worker.mjs` checks the access code, origin, request size, message
format, and rate limit before calling `env.AI.run`. The default model is
`@cf/meta/llama-3.1-8b-instruct-fp8`, configurable through `AI_MODEL`.
The AI binding gives the Worker model access without placing a Cloudflare API
token in browser code.

`cloudflare/context.mjs` supplies server-controlled instructions and short
reference notes from the American Academy of Dermatology and NHS. These
describe scope, project facts, and sensible boundaries for medical education.
This is a prompt with curated context, not model training or a search engine.
It cannot look up recent research or verify an answer in real time. Reference
links in the UI are background reading, not evidence that each generated
answer was checked. Review the notes when project behavior changes.

## Privacy and limitations

- Only typed text and up to four previous question/answer turns are sent.
  There is no photo picker, vision model, database query, or automatic personal
  context collection in chat. Requests reject extra fields and image payloads.
- The app does not persist conversations or log message bodies. The browser
  keeps history and the access code in memory; reload clears both. Cloudflare
  processes the text under its own service policies. Do not type identifying
  or sensitive medical information into the chat.
- Answers are rendered as plain text, never executable HTML. System-role
  messages from clients are rejected. Prompt boundaries reduce misuse but do
  not guarantee model compliance or correctness.
- The assistant offers general information, not diagnosis or prescriptions.
  It should direct serious or persistent concerns to a professional. The spot
  counters are still placeholders; the assistant has no actual counts.
- Use HTTPS for the hosted chat. For local chat, use localhost on the laptop;
  the existing plain-HTTP phone upload route is not appropriate for sending a
  private chat code over an untrusted network.

## Deploy on Cloudflare

Requires Node.js/npm and a Cloudflare account. No new domain is needed: start
with the supplied `workers.dev` address. An existing Cloudflare domain can
later use a dedicated subdomain, without replacing the personal website.

From the repository root:

```powershell
npx wrangler@4 login
npx wrangler@4 deploy --config cloudflare/wrangler.jsonc
npx wrangler@4 secret put CHAT_ACCESS_KEY --config cloudflare/wrangler.jsonc
```

Complete the login in your browser. When the secret command prompts, enter a
long random private chat access code. Never commit that value or put a
Cloudflare account token into the chat page. The initial deployment refuses
AI requests until the secret exists. Share the chat code only with intended
users; it grants chatbot access, not Cloudflare account access. Changing the
secret revokes the old code. This shared code is adequate for a private demo;
a public unrestricted launch needs stronger abuse controls and user access.

Open the printed HTTPS Worker address and enter that code under **Chat access
code**. Test a project question and a general skincare question. Also check
that it declines unrelated tasks, does not invent today's count, and does not
diagnose a described lesion. These checks must use the live model: unit tests
with a mocked model do not establish answer quality.

To connect the laptop upload page, set the address before starting its server:

```powershell
$env:CHAT_SERVICE_URL = 'https://lesion-atlas-chat.YOUR-SUBDOMAIN.workers.dev'
.\.venv\Scripts\python.exe -m src.server.main
```

This setting applies to that shell/process. An always-on server needs the same
environment setting in its startup configuration and a restart. Do not expose
the ingest server through a public tunnel. Deploying the chatbot does not
deploy photo upload, detection, or the local database.

## Free tier and errors

Stay on the Workers Free plan; this implementation does not opt into paid
usage or a paid fallback. Workers AI and Workers each have free usage limits.
The Worker limits responses to 450 output tokens and accepts at most 2,000
characters per message / 10,000 across the request. A Cloudflare rate-limit
binding allows eight requests per minute per Cloudflare location for the
shared application key. This is approximate distributed throttling, not a
hard global daily budget. The private access code is still necessary.

If the free allowance is exhausted or the provider fails, the UI explains
that AI is unavailable. Missing configuration, incorrect access codes, busy
service, and oversized messages also produce plain-English errors. Status
means the bindings exist, not that a live inference was successful.

Check the current provider documentation before changing plans or models:

- [Workers AI pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/)
- [Workers AI bindings](https://developers.cloudflare.com/workers-ai/configuration/bindings/)
- [Workers static assets](https://developers.cloudflare.com/workers/static-assets/)
- [Rate limiting binding](https://developers.cloudflare.com/workers/runtime-apis/bindings/rate-limit/)

## Files and verification

| File | Responsibility |
|---|---|
| `cloudflare/wrangler.jsonc` | Worker, assets, model and rate-limit bindings |
| `cloudflare/worker.mjs` | Validation, authorization, inference, safe errors |
| `cloudflare/context.mjs` | Project facts, scope, reference notes |
| `cloudflare/public/` | Chat interface, memory-only history, plain-text rendering |
| `src/server/chat.py` | Local text-only relay and chat static mount |
| `tests/chat_worker.test.mjs` | Worker boundary and failure tests using a model stub |
| `tests/test_chat.py` | Local relay validation and privacy boundary tests |

Run the automated checks:

```powershell
node --test tests/chat_worker.test.mjs
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
.\.venv\Scripts\python.exe tests/test_tiling.py
.\.venv\Scripts\python.exe tests/test_splits.py
```

Deployment, live model quality, and quota exhaustion require separate checks
against the configured Cloudflare service. A passing local test suite alone
does not mean the cloud deployment is running.
