'use strict';

/**
 * Throwaway smoke test for metadata-aware retrieval in rag.js.
 * Run: node scripts/smoke-metadata-retrieval.js
 *
 * Two tiny documents are seeded straight into the index, mirroring what
 * indexFile produces under v3 rules (passages embedded with their document's
 * caption, number and keywords in front of the page text):
 *
 * - scan.pdf: an ordinary readable page whose wording shares nothing with the
 *   questions asked. It can only rank well through its caption and keywords.
 * - text.pdf: a competing page with no relevant wording or metadata.
 *
 * If the metadata pipeline is broken, scan.pdf loses to (or ties with)
 * text.pdf and every assertion below fails.
 */

const assert = require('assert');
const http = require('http');
const Database = require('better-sqlite3');
const { createRag, buildChunkEmbedText, buildMetadataFtsText, isReadableText } = require('../rag');

function makeDb() {
  const db = new Database(':memory:');
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

/** Same deterministic, length-separable embeddings as the answer-cache smoke test. */
function fakeEmbedding(text) {
  const vector = new Float32Array(8).fill(0.5);
  vector[String(text).length % 8] = 1;
  return vector;
}

function normalize(vector) {
  let sum = 0;
  for (const value of vector) sum += value * value;
  const magnitude = Math.sqrt(sum) || 1;
  return Array.from(vector, value => value / magnitude);
}

/** Fake Ollama: /api/embed mirrors fakeEmbedding, /api/chat streams a canned answer. */
function startFakeOllama() {
  const chatCalls = [];
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
        const embeddings = texts.map(text => Array.from(fakeEmbedding(text)));
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ embeddings }));
      }
      if (req.url === '/api/chat') {
        chatCalls.push(JSON.parse(raw));
        res.writeHead(200, { 'Content-Type': 'application/x-ndjson' });
        res.write(`${JSON.stringify({ message: { role: 'assistant', content: 'Hello ' }, done: false })}\n`);
        return res.end(`${JSON.stringify({ message: { role: 'assistant', content: 'from the model.' }, done: true, done_reason: 'stop' })}\n`);
      }
      res.writeHead(404);
      res.end('{}');
    });
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port, chatCalls })));
}

async function main() {
  process.env.AI_PROVIDER = 'ollama';
  process.env.OLLAMA_CHAT_MODEL = 'test-chat';
  process.env.OLLAMA_EMBED_MODEL = 'test-embed';
  process.env.OLLAMA_EMBED_DOC_PREFIX = '';
  process.env.OLLAMA_EMBED_QUERY_PREFIX = '';
  process.env.RAG_RESCAN_INTERVAL_MS = '60000';

  const { server, port, chatCalls } = await startFakeOllama();
  process.env.OLLAMA_BASE_URL = `http://127.0.0.1:${port}`;

  const db = makeDb();
  db.prepare("INSERT INTO document_types (name, requires_login) VALUES ('SMI', 0)").run();
  db.prepare("INSERT INTO documents (filename, original_name, document_number, document_type, caption, keywords) VALUES ('scan.pdf', 'scan.pdf', 'SMI-900', 'SMI', 'Power transformer maintenance manual', 'transformer, oil, winding, temperature')").run();
  db.prepare("INSERT INTO documents (filename, original_name, document_number, document_type, caption, keywords) VALUES ('text.pdf', 'text.pdf', 'SMI-901', 'SMI', 'Ordinary text document', '')").run();

  const rag = createRag({ db, uploadDirectory: '.', logger: { log: () => {}, warn: () => {}, error: () => {} } });

  // 1. Embedding text carries the metadata in front of the page text.
  const scanMeta = { title: 'Power transformer maintenance manual', documentNumber: 'SMI-900', keywords: 'transformer, oil, winding, temperature' };
  const embedText = buildChunkEmbedText(scanMeta, 'Page words.');
  assert.strictEqual(
    embedText,
    'Document number: SMI-900\nTitle: Power transformer maintenance manual\nKeywords: transformer, oil, winding, temperature\nPage words.'
  );
  // 2. The metadata FTS text holds the searchable terms of every field.
  const ftsText = buildMetadataFtsText(scanMeta);
  assert.ok(/\b900\b/.test(ftsText) && /\btransformer\b/.test(ftsText) && /\bwinding\b/.test(ftsText));

  // Seed the index the way indexFile would: passages embedded with the
  // metadata prefix, document rows stamped with the v3 revision.
  const pages = {
    'scan.pdf': 'This scanned page holds generic workshop notes and weather remarks.',
    'text.pdf': 'Bearing caps are checked with feeler gauges during routine exams.'
  };
  for (const [source, content] of Object.entries(pages)) {
    assert.ok(isReadableText(content), 'seed content must pass the readability gate');
    const doc = db.prepare('SELECT id, caption, document_number, keywords FROM documents WHERE filename = ?').get(source);
    const meta = { title: doc.caption, documentNumber: doc.document_number, keywords: doc.keywords };
    const vector = fakeEmbedding(buildChunkEmbedText(meta, content));
    db.prepare(`INSERT INTO rag_chunks (source, document_id, page, chunk_index, content, embedding, embedding_model, dimensions)
                VALUES (?, ?, 1, 0, ?, ?, 'ollama:test-embed', 8)`)
      .run(source, doc.id, content, Buffer.from(new Float32Array(normalize(vector)).buffer));
    db.prepare(`INSERT INTO rag_documents (source, document_id, title, document_number, document_type, size, modified_ms, page_count, chunk_count, content_chars, status, error, embedding_model, ocr_pages, keywords)
                VALUES (?, ?, ?, ?, 'SMI', 0, 0, 1, 1, 0, 'indexed', '', 'ollama:test-embed|v3', 0, ?)`)
      .run(source, doc.id, doc.caption, doc.document_number, doc.keywords);
  }

  // Build the metadata keyword rows through the public path under test.
  assert.strictEqual(rag.updateDocumentMetadata('scan.pdf'), true);
  assert.strictEqual(rag.updateDocumentMetadata('text.pdf'), true);

  // 3. The metadata row is queryable on its own.
  const metaHit = db.prepare("SELECT rag_key FROM rag_documents_meta_fts WHERE rag_documents_meta_fts MATCH 'transformer'").all();
  assert.deepStrictEqual(metaHit.map(row => row.rag_key), ['scan.pdf'], 'the caption/keywords row must be searchable');

  // 4. A question phrased like the caption retrieves and leads with the
  //    scanned document, whose page text shares no wording with the question.
  const byCaption = await rag.ask('How do I maintain the power transformer?');
  assert.ok(byCaption.sources.length > 0, 'metadata must retrieve the document');
  assert.strictEqual(byCaption.sources[0].source, 'scan.pdf', 'the metadata-matched document must lead the ranking');

  // 5. The passage headers shown to the model carry the document's keywords.
  const prompt = chatCalls.at(-1).messages.at(-1).content;
  assert.match(prompt, /Keywords: transformer, oil, winding, temperature/);

  // 6. A document-number question works even when the number appears nowhere
  //    in the page text — the metadata index matches it.
  const byNumber = await rag.ask('What is SMI-900?');
  assert.ok(byNumber.sources.some(source => source.source === 'scan.pdf'), 'the document number must retrieve the document');
  assert.strictEqual(byNumber.sources[0].source, 'scan.pdf');

  // 7. A caption/keywords edit takes effect immediately, without any PDF
  //    re-reading or re-embedding.
  db.prepare("UPDATE documents SET caption = 'Traction transformer oil care handbook', keywords = 'transformer, oil, desiccant' WHERE filename = 'scan.pdf'").run();
  assert.strictEqual(rag.updateDocumentMetadata('scan.pdf'), true);
  const oldRow = db.prepare("SELECT rag_key FROM rag_documents_meta_fts WHERE rag_documents_meta_fts MATCH 'winding'").all();
  assert.deepStrictEqual(oldRow, [], 'the replaced keywords must be gone from the metadata index');
  const newRow = db.prepare("SELECT rag_key FROM rag_documents_meta_fts WHERE rag_documents_meta_fts MATCH 'desiccant'").all();
  assert.deepStrictEqual(newRow.map(row => row.rag_key), ['scan.pdf'], 'the new keywords must be searchable');
  const afterEdit = await rag.ask('What does the oil care handbook say?');
  assert.strictEqual(afterEdit.sources[0].source, 'scan.pdf', 'the edited caption must drive retrieval right away');

  // 8. Unknown files are reported and change nothing.
  assert.strictEqual(rag.updateDocumentMetadata('missing.pdf'), false);

  server.close();
  console.log('metadata retrieval smoke test: all assertions passed');
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
