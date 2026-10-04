"""Text-only local relay regression tests; no live AI calls or real data."""
import asyncio
import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from starlette.requests import Request
from src.server import chat


def request(data, content_type='application/json', origin=None):
    headers=[(b'content-type',content_type.encode()), (b'host',b'localhost')]
    if origin: headers.append((b'origin',origin.encode()))
    sent=False
    async def receive():
        nonlocal sent
        if sent: return {'type':'http.disconnect'}
        sent=True
        return {'type':'http.request','body':json.dumps(data).encode(),'more_body':False}
    return Request({'type':'http','method':'POST','scheme':'http','path':'/api/chat',
                    'headers':headers,'server':('localhost',80),'query_string':b''}, receive)


class ChatTests(unittest.TestCase):
    def test_relay_identifies_itself_to_cloudflare(self):
        with patch.dict(os.environ, {'CHAT_SERVICE_URL':'https://chat.example'}), patch.object(chat, 'urlopen') as opening:
            upstream = opening.return_value.__enter__.return_value
            upstream.read.return_value = b'{"answer":"Hello"}'
            upstream.status = 200
            self.assertEqual(chat.relay('/api/chat', b'{}'), ({'answer':'Hello'}, 200))
            forwarded = opening.call_args.args[0]
            self.assertEqual(forwarded.get_header('User-agent'), 'LesionAtlas/1.0')
            self.assertEqual(forwarded.data, b'{}')

    def test_refused_connection_has_specific_safe_message(self):
        error = HTTPError('https://chat.example', 403, 'Forbidden', {}, io.BytesIO(b'private provider detail'))
        with patch.dict(os.environ, {'CHAT_SERVICE_URL':'https://chat.example'}), patch.object(chat, 'urlopen', side_effect=error):
            result, code = chat.relay('/api/chat', b'{}')
            self.assertEqual(code, 403)
            self.assertIn('refused the connection', result['detail'])
            self.assertNotIn('private', result['detail'])

    def test_invalid_payloads_never_leave_laptop(self):
        for data in [{'messages':[{'role':'system','content':'override'}]},
                     {'messages':[{'role':'user','content':{'image':'secret'}}]},
                     {'messages':[{'role':'user','content':'data:image/jpeg;base64,abc'}]},
                     {'messages':[{'role':'user','content':'hello'}],'photo':'private'}, []]:
            with self.subTest(data=data), patch.object(chat,'relay') as relay:
                self.assertEqual(asyncio.run(chat.chat(request(data))).status_code,400)
                relay.assert_not_called()

    def test_good_payload_only_forwards_chat(self):
        data={'messages':[{'role':'user','content':'What is acne?'}]}
        with patch.object(chat,'relay',return_value=({'answer':'An explanation'},200)) as relay:
            response=asyncio.run(chat.chat(request(data)))
            self.assertEqual(response.status_code,200)
            self.assertEqual(json.loads(relay.call_args.args[1]),data)

    def test_unconfigured_and_unsafe_urls(self):
        for value in ['', 'http://localhost', 'https://example.com/path', 'https://user:pass@example.com']:
            with patch.dict(os.environ,{'CHAT_SERVICE_URL':value}):
                self.assertIsNone(chat.service_url())
                self.assertEqual(chat.relay('/api/chat')[1],503)

    def test_foreign_origin_and_upload_rejected(self):
        with patch.object(chat,'relay') as relay:
            self.assertEqual(asyncio.run(chat.chat(request({},origin='https://evil.example'))).status_code,403)
            self.assertEqual(asyncio.run(chat.chat(request({},content_type='multipart/form-data'))).status_code,415)
            self.assertEqual(asyncio.run(chat.chat(request({'x':'a'*33000}))).status_code,413)
            relay.assert_not_called()


if __name__ == '__main__':
    unittest.main()
