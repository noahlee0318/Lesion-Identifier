"""Text-only relay to the configured Cloudflare Worker. No image/DB imports."""
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request as URLRequest, urlopen

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

router = APIRouter()
PUBLIC = Path(__file__).resolve().parents[2] / 'cloudflare' / 'public'


def service_url():
    url = os.environ.get('CHAT_SERVICE_URL', '').rstrip('/')
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/'):
        return None
    return url


def relay(path, payload=None):
    base = service_url()
    if not base:
        return {'detail': 'Chat is not connected yet. Set CHAT_SERVICE_URL to your Cloudflare chatbot address.'}, 503
    # Cloudflare rejects urllib's default user agent (403/1010). Identify our
    # actual client explicitly, without impersonating a browser.
    headers = {'Content-Type': 'application/json', 'User-Agent': 'LesionAtlas/1.0'}
    req = URLRequest(base + path, data=payload, headers=headers)
    try:
        with urlopen(req, timeout=35) as response:
            result = json.loads(response.read(64000))
            return result, response.status
    except HTTPError as error:
        messages = {403:'The AI service refused the connection. The website connection needs checking.',
                    429:'Chat is busy. Wait a minute and try again.',
                    400:'Use a short text question. No images or attachments.', 413:'That message is too long.'}
        return {'detail': messages.get(error.code, 'AI is unavailable right now. Check Cloudflare setup or try again later.')}, error.code if error.code in messages else 503
    except (URLError, TimeoutError, ValueError, OSError):
        return {'detail':'Cannot reach the AI service. Try again later.'}, 503


@router.get('/api/chat/status')
async def status():
    if not service_url():
        return {'ready': False}
    data, code = await run_in_threadpool(relay, '/api/chat/status')
    return {'ready': code == 200 and data.get('ready') is True}


@router.post('/api/chat')
async def chat(request: Request):
    origin = request.headers.get('origin')
    if origin and origin.rstrip('/') != str(request.base_url).rstrip('/'):
        return JSONResponse({'detail':'Open chat on this website.'}, status_code=403)
    if not request.headers.get('content-type', '').startswith('application/json'):
        return JSONResponse({'detail':'Only text chat is supported.'}, status_code=415)
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 32000:
            return JSONResponse({'detail':'That message is too long.'}, status_code=413)
    try:
        body = json.loads(raw)
        messages = body['messages']
        if set(body) != {'messages'} or not isinstance(messages, list) or not 1 <= len(messages) <= 9:
            raise ValueError()
        for i, m in enumerate(messages):
            if set(m) != {'role', 'content'} or m['role'] != ('user' if i % 2 == 0 else 'assistant') or not isinstance(m['content'], str) or not 1 <= len(m['content'].strip()) <= 2000:
                raise ValueError()
            if 'data:' in m['content'].lower() or 'base64' in m['content'].lower():
                raise ValueError()
        if messages[-1]['role'] != 'user' or sum(len(m['content']) for m in messages) > 10000:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        return JSONResponse({'detail':'Use a short text question. Images and attachments are not supported.'}, status_code=400)
    data, code = await run_in_threadpool(relay, '/api/chat', json.dumps(body).encode())
    return JSONResponse(data, status_code=code, headers={'Cache-Control':'no-store'})


def install(app):
    app.include_router(router)
    app.mount('/chat', StaticFiles(directory=PUBLIC, html=True), name='chat')
