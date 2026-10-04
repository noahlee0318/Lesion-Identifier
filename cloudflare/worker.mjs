import {systemPrompt, sources} from './context.mjs';

export function validate(body) {
  if (!body || Object.keys(body).length !== 1 || !Array.isArray(body.messages) ||
      body.messages.length < 1 || body.messages.length > 9) throw new Error('Send a short text question.');
  let total = 0;
  const messages = body.messages.map((m, i) => {
    if (!m || Object.keys(m).sort().join() !== 'content,role' ||
        m.role !== (i % 2 === 0 ? 'user' : 'assistant') || typeof m.content !== 'string' ||
        !m.content.trim() || m.content.length > 2000 || /data:|base64|https?:\/\/\S+\.(?:png|jpg|jpeg|webp)/i.test(m.content))
      throw new Error('Use text only, up to 2,000 characters per message. No images or attachments.');
    total += m.content.length;
    return {role: m.role, content: m.content.trim()};
  });
  if (messages.at(-1).role !== 'user' || total > 10000) throw new Error('Start a new chat or shorten your question.');
  return messages;
}
function json(body, status = 200) {
  return Response.json(body, {status, headers: {'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'}});
}
export default {
  async fetch(request, env) {
    const path = new URL(request.url).pathname;
    if (!path.startsWith('/api/')) return env.ASSETS.fetch(request);
    if (path === '/api/chat/status' && request.method === 'GET')
      return json({ready: Boolean(env.AI && env.CHAT_ACCESS_KEY && env.CHAT_LIMIT)});
    if (path !== '/api/chat' || request.method !== 'POST') return json({detail:'Not found.'}, 404);
    const origin = request.headers.get('Origin');
    if (origin && origin !== new URL(request.url).origin) return json({detail:'Open chat on this website.'}, 403);
    if (!env.AI || !env.CHAT_ACCESS_KEY || !env.CHAT_LIMIT) return json({detail:'Chat is not connected yet. Finish the Cloudflare setup first.'}, 503);
    if (request.headers.get('Authorization') !== `Bearer ${env.CHAT_ACCESS_KEY}`)
      return json({detail:'Enter the chat access code to continue.'}, 401);
    const {success} = await env.CHAT_LIMIT.limit({key:'lesion-atlas-chat'});
    if (!success) return json({detail:'Chat is busy. Wait a minute and try again.'}, 429);
    if (!(request.headers.get('Content-Type') || '').startsWith('application/json'))
      return json({detail:'Only text chat is supported.'}, 415);
    let messages;
    try {
      // Bound the stream before parsing; attachments never reach the model.
      const reader = request.body?.getReader();
      if (!reader) return json({detail:'Enter a question.'}, 400);
      let size = 0; const chunks = [];
      while (true) {
        const {done, value} = await reader.read(); if (done) break;
        size += value.length;
        if (size > 32000) { await reader.cancel(); return json({detail:'That message is too long.'}, 413); }
        chunks.push(value);
      }
      const bytes = new Uint8Array(size); let offset = 0;
      for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
      messages = validate(JSON.parse(new TextDecoder().decode(bytes)));
    } catch { return json({detail:'Use a short text question. Images and attachments are not supported.'}, 400); }
    try {
      const result = await env.AI.run(env.AI_MODEL || '@cf/meta/llama-3.1-8b-instruct-fp8', {
        messages:[{role:'system', content:systemPrompt}, ...messages], max_tokens:450, temperature:0.3
      });
      if (!result || typeof result.response !== 'string' || !result.response.trim()) throw new Error('Empty result');
      return json({answer:result.response.trim().slice(0, 12000), sources});
    } catch {
      return json({detail:'AI is unavailable right now. Its free allowance may be used up. Try again later.'}, 503);
    }
  }
};
