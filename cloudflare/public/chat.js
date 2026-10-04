const $ = id => document.getElementById(id);
let history = [];
let busy = false;
function addMessage(role, content) {
  const box = document.createElement('div'); box.className = `message ${role}`;
  const label = document.createElement('strong'); label.textContent = role === 'user' ? 'You' : 'Lesion Atlas';
  box.append(label, document.createTextNode(content)); // Never interpret model output as HTML.
  $('conversation').append(box); box.scrollIntoView({block:'nearest'});
}
async function status() {
  try {
    const response = await fetch('/api/chat/status', {cache:'no-store'});
    const data = await response.json();
    $('connection').textContent = data.ready ? 'AI connected · enter your access code to chat' : 'AI setup pending · connect Cloudflare to enable replies';
  } catch { $('connection').textContent = 'Cannot reach chat right now. Try reloading.'; }
}
document.querySelectorAll('.suggestions button').forEach(button => button.addEventListener('click', () => {
  $('question').value = button.textContent; $('question').focus();
}));
$('clear').addEventListener('click', () => {
  if (busy) return;
  history = []; $('conversation').replaceChildren(); $('chatError').textContent = ''; $('question').value = '';
});
$('chatForm').addEventListener('submit', async event => {
  event.preventDefault(); if (busy) return;
  const question = $('question').value.trim(); if (!question) return;
  const key = $('accessCode').value.trim();
  if (!key) { $('access').open = true; $('accessCode').focus(); $('chatError').textContent = 'Enter your chat access code first.'; return; }
  // Up to four completed turns, bounded further by the server.
  let recent = history.slice(-8);
  while (recent.reduce((n,m) => n + m.content.length, question.length) > 10000) recent = recent.slice(2);
  const messages = [...recent, {role:'user', content:question}];
  busy = true; $('send').disabled = $('clear').disabled = true; $('send').textContent = 'Thinking…'; $('chatError').textContent = '';
  const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), 45000);
  let failureText = 'Cannot reach chat right now. Your question is still here; try again.';
  try {
    const response = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json', Authorization:`Bearer ${key}`}, body:JSON.stringify({messages}), signal:controller.signal});
    const data = await response.json();
    if (!response.ok) {
      failureText = typeof data.detail === 'string' ? data.detail : 'Chat could not answer. Try again.';
      throw new Error('Request failed');
    }
    if (typeof data.answer !== 'string' || !data.answer.trim()) {
      failureText = 'Chat returned an empty answer. Try again.';
      throw new Error('Empty answer');
    }
    addMessage('user', question); addMessage('assistant', data.answer);
    history = [...messages, {role:'assistant', content:data.answer.slice(0,2000)}]; $('question').value = '';
  } catch (error) {
    $('chatError').textContent = error.name === 'AbortError' ? 'That took too long. Your question is still here; try again.' : failureText;
  } finally { clearTimeout(timer); busy = false; $('send').disabled = $('clear').disabled = false; $('send').textContent = 'Send question'; }
});
status();
