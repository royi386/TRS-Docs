'use strict';

/* Temporary regression test: the hosted Gemini provider after the refactor. */

const fs = require('fs');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');

const ROOT = __dirname;
const WORK = path.join(ROOT, '.tmp-gemini-data');
const MOCK_PORT = 4351;
const PORT = 3992;

const seen = { embed: [], chat: [] };

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
    const payload = JSON.parse(body || '{}');
    res.setHeader('Content-Type', 'application/json');
    if (req.url.includes(':batchEmbedContents')) {
      seen.embed.push(payload);
      res.end(JSON.stringify({ embeddings: payload.requests.map(r => ({ values: fakeEmbedding(r.content.parts[0].text) })) }));
      return;
    }
    if (req.url.includes(':generateContent')) {
      seen.chat.push(payload);
      res.end(JSON.stringify({ candidates: [{ content: { parts: [{ text: 'Check it with a dial gauge [1].' }] }, finishReason: 'STOP' }] }));
      return;
    }
    res.statusCode = 404;
    res.end(JSON.stringify({ error: { message: 'unexpected' } }));
  });
});

async function main() {
  fs.rmSync(WORK, { recursive: true, force: true });
  fs.mkdirSync(path.join(WORK, 'uploads'), { recursive: true });
  await new Promise(resolve => mock.listen(MOCK_PORT, '127.0.0.1', resolve));

  const child = spawn(process.execPath, ['server.js'], {
    cwd: ROOT,
    env: {
      ...process.env,
      PORT: String(PORT),
      ADMIN_PASSWORD: 'pw',
      DB_PATH: path.join(WORK, 'test.db'),
      UPLOAD_DIR: path.join(WORK, 'uploads'),
      FEEDBACK_UPLOAD_DIR: path.join(WORK, 'feedback'),
      AI_PROVIDER: 'gemini',
      GEMINI_API_KEY: 'test-key',
      GEMINI_API_BASE: `http://127.0.0.1:${MOCK_PORT}/v1beta`,
      GEMINI_EMBED_DIMENSIONS: '8',
      RAG_EMBED_DELAY_MS: '0'
    },
    stdio: ['ignore', 'pipe', 'pipe']
  });
  const log = [];
  child.stdout.on('data', d => log.push(String(d)));
  child.stderr.on('data', d => log.push(String(d)));

  const base = `http://127.0.0.1:${PORT}`;
  try {
    let up = false;
    for (let i = 0; i < 60 && !up; i += 1) {
      try { up = (await fetch(`${base}/api/health`)).ok; } catch { /* not up yet */ }
      if (!up) await new Promise(r => setTimeout(r, 300));
    }
    check('server boots', up);

    const status = await (await fetch(`${base}/api/chat/status`)).json();
    check('status reports the hosted provider', status.provider === 'gemini', status.provider);
    check('status reports it enabled and online', status.enabled === true && status.online === true);
    check('applies the hosted context budget', true);

    const { token } = await (await fetch(`${base}/api/admin/login`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ password: 'pw' })
    })).json();

    const form = new FormData();
    form.append('pdf', new Blob([buildPdf(['RDSO SPECIFICATION No. SMI/EL/1234', 'Bearing clearance is measured with a dial gauge.'])], { type: 'application/pdf' }), 'spec.pdf');
    form.append('documentType', 'SMI');
    form.append('documentNumber', 'SMI/EL/1234');
    form.append('caption', 'Axle bearing inspection');
    check('upload succeeds', (await fetch(`${base}/api/upload`, { method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: form })).ok);

    let after = status;
    for (let i = 0; i < 40 && after.indexedFiles < 1; i += 1) {
      await new Promise(r => setTimeout(r, 300));
      after = await (await fetch(`${base}/api/chat/status`)).json();
    }
    check('indexes via the hosted embeddings', after.indexedFiles === 1, `indexedFiles=${after.indexedFiles}`);
    check('sends the embedding task type', seen.embed[0].requests[0].taskType === 'RETRIEVAL_DOCUMENT', seen.embed[0].requests[0].taskType);
    check('sends the embedding dimensions', seen.embed[0].requests[0].outputDimensionality === 8);

    const chatResponse = await fetch(`${base}/api/chat`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: 'How is bearing clearance measured?' })
    });
    const chat = await chatResponse.json();
    check('answers via the hosted model', chatResponse.ok && /dial gauge/.test(chat.answer || ''), (chat.answer || '').slice(0, 40));
    check('returns sources', chat.sources?.[0]?.documentNumber === 'SMI/EL/1234');
    check('grounds the prompt', seen.chat[0].contents[0].parts[0].text.includes('SMI/EL/1234'));
    check('keeps the system instruction', Boolean(seen.chat[0].systemInstruction));
    check('asks for low thinking on gemini 3', seen.chat[0].generationConfig?.thinkingLevel === 'low', JSON.stringify(seen.chat[0].generationConfig));
    check('uses the query task type for retrieval', seen.embed[seen.embed.length - 1].requests[0].taskType === 'RETRIEVAL_QUERY');
  } finally {
    child.kill();
    mock.close();
    await new Promise(r => setTimeout(r, 400));
    if (process.exitCode) console.log('\n--- server log ---\n' + log.join(''));
    fs.rmSync(WORK, { recursive: true, force: true });
  }
  console.log('\nHosted provider regression check finished.');
}

main().catch(error => { console.error('GEMINI E2E ERROR:', error); process.exitCode = 1; });
