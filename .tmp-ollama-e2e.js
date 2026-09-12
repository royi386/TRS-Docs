'use strict';

/*
 * Temporary test for the local Ollama provider.
 * Boots the real server against a mock Ollama server and checks indexing,
 * request shapes, answer parsing and the failure messages.
 */

const fs = require('fs');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');

const ROOT = __dirname;
const WORK = path.join(ROOT, '.tmp-ollama-data');
const UPLOADS = path.join(WORK, 'uploads');
const MOCK_PORT = 4331;
const DEAD_PORT = 4341;

// 'ready' = both models present, 'no-models' = nothing pulled,
// 'no-chat' = the embed model is present but the chat model is missing.
let mode = 'ready';
const EMBED_MODEL = 'nomic-embed-text';
const CHAT_MODEL = 'qwen3:1.7b';
const seen = { embed: [], chat: [], tags: 0 };

function notFound(model) {
  return { status: 404, body: { error: `model "${model}" not found, try pulling it first` } };
}

function check(label, condition, detail = '') {
  console.log(`${condition ? 'PASS' : 'FAIL'}  ${label}${detail ? ` :: ${detail}` : ''}`);
  if (!condition) process.exitCode = 1;
}

function buildPdf(lines) {
  const objects = [];
  objects[1] = '<< /Type /Catalog /Pages 2 0 R >>';
  objects[2] = '<< /Type /Pages /Kids [4 0 R] /Count 1 >>';
  objects[3] = '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>';
  objects[4] = '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>';
  const stream = ['BT', '/F1 12 Tf', '14 TL', '50 750 Td']
    .concat(lines.map((line, index) => `${index === 0 ? '' : 'T* '}(${line}) Tj`)).concat(['ET']).join('\n');
  objects[5] = `<< /Length ${Buffer.byteLength(stream, 'latin1')} >>\nstream\n${stream}\nendstream`;
  let pdf = '%PDF-1.4\n';
  const offsets = [];
  for (let number = 1; number <= 5; number += 1) {
    offsets[number] = Buffer.byteLength(pdf, 'latin1');
    pdf += `${number} 0 obj\n${objects[number]}\nendobj\n`;
  }
  const xrefOffset = Buffer.byteLength(pdf, 'latin1');
  pdf += 'xref\n0 6\n0000000000 65535 f \n';
  for (let number = 1; number <= 5; number += 1) pdf += `${String(offsets[number]).padStart(10, '0')} 00000 n \n`;
  pdf += `trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`;
  return Buffer.from(pdf, 'latin1');
}

function fakeEmbedding(text, dimensions = 8) {
  const vector = new Array(dimensions).fill(0);
  for (const token of String(text).toLowerCase().match(/[a-z0-9]+/g) || []) {
    let hash = 0;
    for (let index = 0; index < token.length; index += 1) hash = (hash * 31 + token.charCodeAt(index)) >>> 0;
    vector[hash % dimensions] += 1;
  }
  if (!vector.some(value => value > 0)) vector[0] = 1;
  return vector;
}

const mock = http.createServer((req, res) => {
  let body = '';
  req.on('data', chunk => { body += chunk; });
  req.on('end', () => {
    res.setHeader('Content-Type', 'application/json');

    if (req.url === '/api/tags' && req.method === 'GET') {
      seen.tags += 1;
      const models = mode === 'ready'
        ? [{ name: `${EMBED_MODEL}:latest` }, { name: CHAT_MODEL }]
        : mode === 'no-chat'
          ? [{ name: `${EMBED_MODEL}:latest` }]
          : [{ name: 'some-other-model:latest' }];
      res.end(JSON.stringify({ models }));
      return;
    }

    const payload = JSON.parse(body || '{}');
    if (req.url === '/api/embed') {
      if (mode === 'no-models') {
        const failure = notFound(EMBED_MODEL);
        res.statusCode = failure.status;
        res.end(JSON.stringify(failure.body));
        return;
      }
      seen.embed.push(payload);
      res.end(JSON.stringify({ embeddings: payload.input.map(text => fakeEmbedding(text)) }));
      return;
    }
    if (req.url === '/api/chat') {
      if (mode !== 'ready') {
        const failure = notFound(CHAT_MODEL);
        res.statusCode = failure.status;
        res.end(JSON.stringify(failure.body));
        return;
      }
      seen.chat.push(payload);
      res.end(JSON.stringify({
        model: payload.model,
        message: { role: 'assistant', content: 'Bearing clearance is checked with a dial gauge [1].', thinking: 'ignored' },
        done: true,
        done_reason: 'stop'
      }));
      return;
    }

    res.statusCode = 404;
    res.end(JSON.stringify({ error: `unexpected ${req.url}` }));
  });
});

async function bootServer({ port, baseUrl, workDir }) {
  fs.mkdirSync(path.join(workDir, 'uploads'), { recursive: true });
  const child = spawn(process.execPath, ['server.js'], {
    cwd: ROOT,
    env: {
      ...process.env,
      PORT: String(port),
      ADMIN_PASSWORD: 'pw',
      DB_PATH: path.join(workDir, 'test.db'),
      UPLOAD_DIR: path.join(workDir, 'uploads'),
      FEEDBACK_UPLOAD_DIR: path.join(workDir, 'feedback'),
      AI_PROVIDER: 'ollama',
      OLLAMA_BASE_URL: baseUrl,
      RAG_EMBED_DELAY_MS: '0'
    },
    stdio: ['ignore', 'pipe', 'pipe']
  });
  const log = [];
  child.stdout.on('data', d => log.push(String(d)));
  child.stderr.on('data', d => log.push(String(d)));

  const base = `http://127.0.0.1:${port}`;
  for (let attempt = 0; attempt < 60; attempt += 1) {
    try {
      if ((await fetch(`${base}/api/health`)).ok) return { child, base, log };
    } catch { /* not up yet */ }
    await new Promise(r => setTimeout(r, 300));
  }
  throw new Error('server did not start');
}

async function main() {
  fs.rmSync(WORK, { recursive: true, force: true });
  await new Promise(resolve => mock.listen(MOCK_PORT, '127.0.0.1', resolve));

  const mockUrl = `http://127.0.0.1:${MOCK_PORT}`;

  /* ---------- phase A: everything working ---------- */
  let server = await bootServer({ port: 3996, baseUrl: mockUrl, workDir: path.join(WORK, 'a') });
  try {
    const status = await (await fetch(`${server.base}/api/chat/status`)).json();
    check('status reports the local provider', status.provider === 'ollama', status.provider);
    check('status reports the local server online', status.online === true);
    check('status reports no missing models', Array.isArray(status.missingModels) && status.missingModels.length === 0, JSON.stringify(status.missingModels));
    check('status reports the local model names', status.chatModel === 'qwen3:1.7b' && status.embedModel === 'nomic-embed-text', `${status.chatModel} / ${status.embedModel}`);

    const { token } = await (await fetch(`${server.base}/api/admin/login`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ password: 'pw' })
    })).json();

    const form = new FormData();
    form.append('pdf', new Blob([buildPdf(['RDSO SPECIFICATION No. SMI/EL/1234', 'Bearing clearance is measured with a dial gauge.'])], { type: 'application/pdf' }), 'spec.pdf');
    form.append('documentType', 'SMI');
    form.append('documentNumber', 'SMI/EL/1234');
    form.append('caption', 'Axle bearing inspection');
    check('upload succeeds', (await fetch(`${server.base}/api/upload`, { method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: form })).ok);

    let after = status;
    for (let i = 0; i < 40 && after.indexedFiles < 1; i += 1) {
      await new Promise(r => setTimeout(r, 300));
      after = await (await fetch(`${server.base}/api/chat/status`)).json();
    }
    check('indexes the uploaded PDF', after.indexedFiles === 1, `indexedFiles=${after.indexedFiles}`);
    check('calls the local embedding endpoint', seen.embed.length > 0, `${seen.embed.length} calls`);
    check('applies the document task prefix', seen.embed[0].input.every(text => text.startsWith('search_document: ')), seen.embed[0].input[0].slice(0, 30));

    const chatResponse = await fetch(`${server.base}/api/chat`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: 'How is bearing clearance measured?' })
    });
    const chat = await chatResponse.json();
    check('answers from the local model', chatResponse.ok && /dial gauge/.test(chat.answer || ''), (chat.answer || '').slice(0, 50));
    check('returns sources', chat.sources?.length > 0 && chat.sources[0].documentNumber === 'SMI/EL/1234');

    const chatBody = seen.chat[0];
    check('sends a system and user message', chatBody.messages?.[0]?.role === 'system' && chatBody.messages?.[1]?.role === 'user');
    check('asks for a non streaming reply', chatBody.stream === false);
    check('disables thinking for speed', chatBody.think === false);
    check('bounds the answer length', chatBody.options?.num_predict === 400, JSON.stringify(chatBody.options));
    check('sets a small context window', chatBody.options?.num_ctx === 4096);
    check('applies the query task prefix', seen.embed[seen.embed.length - 1].input[0].startsWith('search_query: '), seen.embed[seen.embed.length - 1].input[0].slice(0, 30));
    check('ignores the thinking field', !/ignored/.test(chat.answer || ''));
  } finally {
    server.child.kill();
    await new Promise(r => setTimeout(r, 400));
  }

  /* ---------- phase B: nothing pulled yet ---------- */
  mode = 'no-models';
  server = await bootServer({ port: 3995, baseUrl: mockUrl, workDir: path.join(WORK, 'b') });
  try {
    const status = await (await fetch(`${server.base}/api/chat/status`)).json();
    check('reports both models as missing', status.missingModels.length === 2, JSON.stringify(status.missingModels));
    check('still reports the provider as reachable', status.online === true && status.enabled === true);
    check('log warns about the missing models', server.log.join('').includes('not downloaded yet'));
  } finally {
    server.child.kill();
    await new Promise(r => setTimeout(r, 400));
  }

  /* ---------- phase C: only the chat model is missing ---------- */
  mode = 'no-chat';
  server = await bootServer({ port: 3993, baseUrl: mockUrl, workDir: path.join(WORK, 'c') });
  try {
    const { token } = await (await fetch(`${server.base}/api/admin/login`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ password: 'pw' })
    })).json();
    const form = new FormData();
    form.append('pdf', new Blob([buildPdf(['RDSO SPECIFICATION No. SMI/EL/1234', 'Bearing clearance is measured with a dial gauge.'])], { type: 'application/pdf' }), 'spec.pdf');
    form.append('documentType', 'SMI');
    form.append('documentNumber', 'SMI/EL/1234');
    form.append('caption', 'Axle bearing inspection');
    await fetch(`${server.base}/api/upload`, { method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: form });

    let status = { indexedFiles: 0 };
    for (let i = 0; i < 40 && status.indexedFiles < 1; i += 1) {
      await new Promise(r => setTimeout(r, 300));
      status = await (await fetch(`${server.base}/api/chat/status`)).json();
    }
    check('content indexes while only the chat model is missing', status.indexedFiles === 1, `indexedFiles=${status.indexedFiles}`);
    check('reports only the chat model as missing', status.missingModels.length === 1 && status.missingModels[0] === 'qwen3:1.7b', JSON.stringify(status.missingModels));

    const chatResponse = await fetch(`${server.base}/api/chat`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: 'How is bearing clearance measured?' })
    });
    const chat = await chatResponse.json();
    check('chat fails with a clear 503', chatResponse.status === 503, `status=${chatResponse.status}`);
    check('the error tells the user how to pull the model', /ollama pull qwen3:1\.7b/.test(chat.error || ''), chat.error);
  } finally {
    server.child.kill();
    await new Promise(r => setTimeout(r, 400));
  }

  /* ---------- phase D: ollama unreachable ---------- */
  mode = 'ready';
  server = await bootServer({ port: 3994, baseUrl: `http://127.0.0.1:${DEAD_PORT}`, workDir: path.join(WORK, 'c') });
  try {
    const status = await (await fetch(`${server.base}/api/chat/status`)).json();
    check('reports the provider offline', status.online === false);
    check('explains how to fix it', /not responding/.test(status.hint || ''), status.hint);
    check('the site still works', (await fetch(`${server.base}/`)).ok);
    check('log warns at start up', server.log.join('').includes('not responding'));

    const chatResponse = await fetch(`${server.base}/api/chat`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: 'hello there' })
    });
    const chat = await chatResponse.json();
    check('chat returns a reachable error', chatResponse.status === 502, `status=${chatResponse.status}`);
    check('the error names the local server', /local AI server/.test(chat.error || ''), chat.error);
  } finally {
    server.child.kill();
    await new Promise(r => setTimeout(r, 400));
  }

  mock.close();
  fs.rmSync(WORK, { recursive: true, force: true });
  console.log('\nLocal provider test finished.');
}

main().catch(error => { console.error('OLLAMA E2E ERROR:', error); process.exitCode = 1; });
