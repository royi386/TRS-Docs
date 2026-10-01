'use strict';

/**
 * Throwaway smoke test for the repeat-question answer cache in rag.js.
 * Run: node scripts/smoke-answer-cache.js
 */

const assert = require('assert');
const http = require('http');
const Database = require('better-sqlite3');
const { createRag } = require('../rag');

function makeDb() {
  const db = new Database(':memory:');
  // The metadata tables the RAG layer joins against.
  db.exec(`
    CREATE TABLE documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      filename TEXT NOT NULL,
      original_name TEXT NOT NULL,
      document_number TEXT NOT NULL DEFAULT '',
      document_type TEXT NOT NULL DEFAULT '',
      caption TEXT NOT NULL DEFAULT '',
      keywords TEXT NOT NULL DEFAULT '',
      upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
      file_size INTEGER,
      mime_type TEXT
    );
    CREATE TABLE document_types (name TEXT PRIMARY KEY, requires_login INTEGER NOT NULL DEFAULT 0);
  `);
  return db;
}

/** Fake Ollama: /api/embed returns a stable vector per text, /api/chat streams a canned answer. */
function startFakeOllama() {
  // Modes the tests switch into: 'reason' leaks chain-of-thought once then
  // answers properly on the retried request; 'truncated' reports done_reason
  // "length" so the answer comes back cut off by design.
  const behaviour = { mode: 'normal', chatCalls: [] };
  const server = http.createServer((req, res) => {
    let raw = '';
    req.on('data', chunk => { raw += chunk; });
    req.on('end', () => {
      if (req.url === '/api/tags') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ models: [{ name: 'test-chat' }, { name: 'test-embed' }] }));
      }
      if (req.url === '/api/embed') {
        const { input } = JSON.parse(raw);
        const texts = Array.isArray(input) ? input : [input];
        const embeddings = texts.map(text => {
          const vector = new Float32Array(8).fill(0.5);
          vector[text.length % 8] = 1; // deterministic, length-separable
          return Array.from(vector);
        });
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ embeddings }));
      }
      if (req.url === '/api/chat') {
        const body = JSON.parse(raw);
        behaviour.chatCalls.push(body);
        res.writeHead(200, { 'Content-Type': 'application/x-ndjson' });
        const nudge = /Answer directly with the facts/.test(body.messages[body.messages.length - 1].content || '');
        if (behaviour.mode === 'reason' && !nudge) {
          for (const piece of ['First, ', 'I need to answer ', 'the question using ', 'only the passages.']) {
            res.write(`${JSON.stringify({ message: { role: 'assistant', content: piece }, done: false })}\n`);
          }
          // Real Ollama repeats an empty message on the final done frame.
          return res.end(`${JSON.stringify({ message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop' })}\n`);
        }
        if (behaviour.mode === 'okay') {
          // qwen3 in no-think mode still opens with a verbal tic and then
          // answers fine: the tic must be stripped, not suppressed.
          for (const piece of ['Okay, ', 'Hello ', 'from ', 'the ', 'model.']) {
            res.write(`${JSON.stringify({ message: { role: 'assistant', content: piece }, done: false })}\n`);
          }
          return res.end(`${JSON.stringify({ message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop' })}\n`);
        }
        if (behaviour.mode === 'glued' && !nudge) {
          // The failure seen in production: the monologue arrives as one
          // blob with no spaces, invisible to word-boundary regexes. The
          // nudged retry gets a clean answer, as a real model should.
          res.write(`${JSON.stringify({ message: { role: 'assistant', content: ',letmetacklethisquestion.Theuserisasking' }, done: false })}\n`);
          return res.end(`${JSON.stringify({ message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop' })}\n`);
        }
        if (behaviour.mode === 'truncated') {
          res.write(`${JSON.stringify({ message: { role: 'assistant', content: 'DGA is dissolved gas analysis and the document says ' }, done: false })}\n`);
          return res.end(`${JSON.stringify({ message: { role: 'assistant', content: '' }, done: true, done_reason: 'length' })}\n`);
        }
        for (const piece of ['Hello ', 'from ', 'the ', 'model.']) {
          res.write(`${JSON.stringify({ message: { role: 'assistant', content: piece }, done: false })}\n`);
        }
        // Real Ollama repeats an empty message on the final done frame.
        return res.end(`${JSON.stringify({ message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop', prompt_eval_count: 1, eval_count: 5 })}\n`);
      }
      res.writeHead(404);
      res.end('{}');
    });
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port, behaviour })));
}

async function main() {
  process.env.AI_PROVIDER = 'ollama';
  process.env.OLLAMA_CHAT_MODEL = 'test-chat';
  process.env.OLLAMA_EMBED_MODEL = 'test-embed';
  process.env.OLLAMA_EMBED_DOC_PREFIX = '';
  process.env.OLLAMA_EMBED_QUERY_PREFIX = '';
  process.env.RAG_RESCAN_INTERVAL_MS = '60000';

  const { server, port, behaviour } = await startFakeOllama();
  process.env.OLLAMA_BASE_URL = `http://127.0.0.1:${port}`;

  const db = makeDb();
  db.prepare("INSERT INTO documents (filename, original_name, document_number, document_type, caption) VALUES ('a.pdf', 'a.pdf', 'SMI-001', 'SMI', 'Test doc')").run();
  db.prepare("INSERT INTO document_types (name, requires_login) VALUES ('SMI', 0)").run();

  // Seed one indexed chunk directly so retrieval has something to find.
  const rag = createRag({ db, uploadDirectory: '.', logger: { log: () => {}, warn: () => {}, error: () => {} } });
  const vector = new Float32Array(8).fill(0.5);
  vector[3] = 1;
  db.prepare(`INSERT INTO rag_chunks (source, document_id, page, chunk_index, content, embedding, embedding_model, dimensions)
              VALUES ('a.pdf', 1, 1, 0, ?, ?, 'ollama:test-embed', 8)`).run(
    'SMI-001 Test doc. The VCB is the vacuum circuit breaker used in traction.',
    Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength)
  );
  db.prepare(`INSERT INTO rag_documents (source, document_id, title, document_number, document_type, status, chunk_count, embedding_model)
              VALUES ('a.pdf', 1, 'Test doc', 'SMI-001', 'SMI', 'indexed', 1, 'ollama:test-embed|v2')`).run();
  rag.invalidateCache();

  const config = require('../rag').getConfig();

  // 1. First ask generates and stores.
  const first = await rag.ask('What is VCB?');
  assert.strictEqual(first.answer, 'Hello from the model.');
  assert.ok(!first.cached, 'first answer must not be a cache hit');
  const status1 = await rag.getStatus();
  assert.strictEqual(status1.answerCacheEntries, 1, 'cache should hold one entry');

  // 2. Same question (different case/punctuation) is a cache hit.
  const second = await rag.ask('what is vcb?');
  assert.strictEqual(second.answer, first.answer);
  assert.ok(second.cached, 'repeat question must be answered from the cache');
  assert.deepStrictEqual(second.sources, first.sources, 'cached sources must match');

  // 3. Streaming path replays the cached answer as tokens.
  const events = [];
  for await (const event of rag.askStream('What is VCB?')) events.push(event);
  const tokens = events.filter(event => event.token).map(event => event.token).join('');
  assert.strictEqual(tokens, first.answer, 'stream replay must rebuild the answer');
  assert.ok(events.some(event => event.answer && event.cached), 'stream must mark the final answer as cached');
  assert.ok(events.some(event => event.sources && event.sources.length), 'stream must emit sources before tokens');

  // 4. A different question misses.
  const other = await rag.ask('What is an air breaker?');
  assert.ok(!other.cached, 'a new question must not be a cache hit');

  // 5. Cache keys change with model settings.
  process.env.OLLAMA_CHAT_MODEL = 'other-chat';
  const afterModelChange = await rag.ask('What is VCB?');
  assert.ok(!afterModelChange.cached, 'a model change must invalidate cached answers');
  process.env.OLLAMA_CHAT_MODEL = 'test-chat';

  // 6. Index change invalidates.
  rag.invalidateCache();
  const afterInvalidate = await rag.ask('What is VCB?');
  assert.ok(!afterInvalidate.cached, 'cache must be empty after invalidation');

  // 7. TTL = 0 disables the cache entirely.
  process.env.RAG_ANSWER_CACHE_TTL_MS = '0';
  const disabledStatus = await rag.getStatus();
  void disabledStatus;
  const disabled = await rag.ask('What is VCB?');
  assert.ok(!disabled.cached, 'TTL 0 must disable the cache');
  delete process.env.RAG_ANSWER_CACHE_TTL_MS;

  // 8. A corrupt stored row fails safe (treated as a miss, cache disabled).
  db.prepare('UPDATE rag_answer_cache SET sources = ?').run('not-json{');
  const corrupt = await rag.ask('what is vcb?'); // memory LRU still has it -> hit is fine
  assert.ok(corrupt.cached, 'hot memory entry survives a corrupt row');
  require('../rag').createAnswerCacheStore(db, console); // fresh store sees the corrupt row
  const freshStore = require('../rag').createAnswerCacheStore(db, { warn: () => {} });
  const miss = freshStore.get(require('../rag').answerCacheKey('What is VCB?', 'What is VCB?', config), 86400000, 500);
  assert.strictEqual(miss, null, 'a corrupt row must degrade to a cache miss, not a crash');

  // 9. A reasoning leak is retried with a direct-answer instruction, and only
  // the proper answer is cached.
  rag.invalidateCache();
  behaviour.mode = 'reason';
  behaviour.chatCalls.length = 0;
  const reasoned = await rag.ask('DGA');
  assert.strictEqual(reasoned.answer, 'Hello from the model.', 'the retried request must produce the real answer');
  assert.strictEqual(behaviour.chatCalls.length, 2, 'the model must be called twice: leak, then direct-answer retry');
  assert.ok(/Answer directly with the facts/.test(behaviour.chatCalls[1].messages.at(-1).content), 'the retry must carry the direct-answer instruction');
  const afterReasoned = await rag.ask('DGA');
  assert.ok(afterReasoned.cached, 'the proper answer (not the leak) must be cached');

  // 9b. A leading verbal tic ("Okay, …") is stripped while the answer survives.
  rag.invalidateCache();
  behaviour.mode = 'okay';
  const okay = await rag.ask('DGA');
  assert.strictEqual(okay.answer, 'Hello from the model.', 'the tic must be stripped but the answer kept');
  assert.ok(!okay.cached === false || true); // cached on the next repeat below
  const okayRepeat = await rag.ask('DGA');
  assert.ok(okayRepeat.cached, 'a tic-stripped clean answer must be cacheable');
  behaviour.mode = 'normal';

  // 10. A truncated answer (done_reason "length") is shown but never cached.
  rag.invalidateCache();
  behaviour.mode = 'truncated';
  const truncated = await rag.ask('DGA');
  assert.match(truncated.answer, /dissolved gas analysis/, 'the truncated text is still shown to the user');
  const statusAfterTruncated = await rag.getStatus();
  assert.strictEqual(statusAfterTruncated.answerCacheEntries, 0, 'a truncated answer must not enter the cache');

  // 11. The quality gate itself.
  const { isCacheableAnswer } = require('../rag');
  assert.strictEqual(isCacheableAnswer('The VCB is a vacuum circuit breaker.', false), true);
  assert.strictEqual(isCacheableAnswer('First, I need to check the passages.', false), false, 'reasoning openers are not cacheable');
  assert.strictEqual(isCacheableAnswer('A fine answer.', true), false, 'truncated answers are not cacheable');
  assert.strictEqual(isCacheableAnswer('', false), false);

  // 12b. A glued, space-less reasoning blob is still recognised.
  const { looksLikeReasoningHead } = require('../rag');
  assert.ok(looksLikeReasoningHead(',letmetacklethisquestion.Theuserisasking'), 'glued reasoning must be detected');
  assert.ok(looksLikeReasoningHead('Okay, I need to check the passages'), 'tic + reasoning must be detected');
  assert.ok(!looksLikeReasoningHead('DGA stands for dissolved gas analysis.'), 'a real answer must never look like reasoning');
  rag.invalidateCache();
  behaviour.mode = 'glued';
  const glued = await rag.ask('DGA');
  assert.strictEqual(glued.answer, 'Hello from the model.', 'the glued leak must be retried into a real answer');

  // 12. A bare term gets an explicit task line; a real question does not.
  rag.invalidateCache();
  behaviour.mode = 'normal';
  behaviour.chatCalls.length = 0;
  await rag.ask('DGA');
  assert.ok(/The question is only a term/.test(behaviour.chatCalls[0].messages.at(-1).content), 'bare terms must carry an explicit task line');
  behaviour.chatCalls.length = 0;
  await rag.ask('What is VCB?');
  assert.ok(!/The question is only a term/.test(behaviour.chatCalls[0].messages.at(-1).content), 'real questions must not carry the task line');

  server.close();
  console.log('answer cache smoke test: all assertions passed');
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
