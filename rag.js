'use strict';

/**
 * Retrieval Augmented Generation for the Rail Docs PDF library.
 *
 * PDFs are read straight from the uploads folder, split into page aware chunks,
 * embedded with Google Gemini and stored in SQLite. Answers are generated only
 * from the retrieved chunks, so the chatbot cannot invent content that is not
 * in the library.
 */

const fs = require('fs');
const path = require('path');

const GEMINI_API_BASE = 'https://generativelanguage.googleapis.com/v1beta';

// Model names move quickly. The first entry is the default, the rest are only
// tried when the API says the model itself is unavailable.
const CHAT_MODEL_FALLBACKS = [
  'gemini-3.8-flash',
  'gemini-3.7-flash',
  'gemini-3.6-flash',
  'gemini-2.5-flash',
  'gemini-2.0-flash'
];

const DEFAULT_EMBED_MODEL = 'gemini-embedding-001';
const DEFAULT_EMBED_DIMENSIONS = 768;
const EMBED_BATCH_MAX = 100;

// Local defaults. qwen3:1.7b and nomic-embed-text together stay inside about
// 1.7 GB of RAM, which matters on a box already running Immich.
const DEFAULT_OLLAMA_BASE_URL = 'http://127.0.0.1:11434';
const DEFAULT_OLLAMA_CHAT_MODEL = 'qwen3:1.7b';
const DEFAULT_OLLAMA_EMBED_MODEL = 'nomic-embed-text';

// A local CPU model needs a far smaller prompt and answer budget than a hosted
// one, or the first token takes minutes to arrive.
const LOCAL_LIMITS = { topK: 4, maxContextChars: 4000, answerMaxTokens: 400, embedBatch: 16, embedDelayMs: 0 };
const CLOUD_LIMITS = { topK: 6, maxContextChars: 12000, answerMaxTokens: 2048, embedBatch: 24, embedDelayMs: 250 };

const SYSTEM_INSTRUCTION = [
  'You are the Rail Docs assistant for an Indian Railways engineering document library.',
  'You answer questions using ONLY the numbered context passages supplied with each question.',
  '',
  'Rules:',
  '1. Use only facts that appear in the context passages. Never use outside knowledge, and never guess.',
  '2. If the context does not contain the answer, reply that the library does not appear to cover it and list the closest related documents you did find. Do not invent document numbers, dates or values.',
  '3. Cite the passages you use with their bracketed numbers, for example [1] or [2][3].',
  '4. Prefer exact document numbers, figures and technical wording from the passages.',
  '5. If the passages disagree, say so rather than picking one silently.',
  '6. Answer in the same language the question was asked in.',
  '7. Be concise and practical. Use short paragraphs or a brief list. Do not repeat the question back.',
  '8. When you mention a document, use its document number and caption from the passage header.'
].join('\n');

function readNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function getConfig() {
  const requested = String(process.env.AI_PROVIDER || '').trim().toLowerCase();
  const provider = requested === 'gemini' || requested === 'ollama' ? requested : 'ollama';
  const limits = provider === 'ollama' ? LOCAL_LIMITS : CLOUD_LIMITS;

  return {
    provider,

    // Local Ollama server
    ollamaBaseUrl: (process.env.OLLAMA_BASE_URL || DEFAULT_OLLAMA_BASE_URL).replace(/\/+$/, ''),
    ollamaChatModel: process.env.OLLAMA_CHAT_MODEL || DEFAULT_OLLAMA_CHAT_MODEL,
    ollamaEmbedModel: process.env.OLLAMA_EMBED_MODEL || DEFAULT_OLLAMA_EMBED_MODEL,
    ollamaNumCtx: readNumber(process.env.OLLAMA_NUM_CTX, 4096),
    ollamaKeepAlive: process.env.OLLAMA_KEEP_ALIVE || '5m',
    // nomic-embed-text is trained with these task prefixes. Set them to an
    // empty string when switching to an embedding model that has none.
    ollamaDocPrefix: process.env.OLLAMA_EMBED_DOC_PREFIX ?? 'search_document: ',
    ollamaQueryPrefix: process.env.OLLAMA_EMBED_QUERY_PREFIX ?? 'search_query: ',
    ollamaTemperature: Number.isFinite(Number(process.env.OLLAMA_TEMPERATURE)) ? Number(process.env.OLLAMA_TEMPERATURE) : 0.2,

    // Hosted Gemini
    apiKey: process.env.GEMINI_API_KEY || '',
    apiBase: (process.env.GEMINI_API_BASE || GEMINI_API_BASE).replace(/\/+$/, ''),
    embedModel: process.env.GEMINI_EMBED_MODEL || DEFAULT_EMBED_MODEL,
    embedDimensions: readNumber(process.env.GEMINI_EMBED_DIMENSIONS, DEFAULT_EMBED_DIMENSIONS),
    chatModel: process.env.GEMINI_CHAT_MODEL || CHAT_MODEL_FALLBACKS[0],

    // Shared
    topK: Math.min(readNumber(process.env.RAG_TOP_K, limits.topK), 20),
    maxContextChars: readNumber(process.env.RAG_MAX_CONTEXT_CHARS, limits.maxContextChars),
    minScore: Number.isFinite(Number(process.env.RAG_MIN_SCORE)) ? Number(process.env.RAG_MIN_SCORE) : 0.3,
    chunkChars: readNumber(process.env.RAG_CHUNK_CHARS, 1400),
    chunkOverlap: readNumber(process.env.RAG_CHUNK_OVERLAP, 200),
    embedBatch: Math.min(readNumber(process.env.RAG_EMBED_BATCH, limits.embedBatch), EMBED_BATCH_MAX),
    embedDelayMs: Number.isFinite(Number(process.env.RAG_EMBED_DELAY_MS)) ? Math.max(Number(process.env.RAG_EMBED_DELAY_MS), 0) : limits.embedDelayMs,
    rescanIntervalMs: readNumber(process.env.RAG_RESCAN_INTERVAL_MS, 5 * 60 * 1000),
    maxCharsPerFile: readNumber(process.env.RAG_MAX_CHARS_PER_FILE, 1500000),
    maxChunksPerFile: readNumber(process.env.RAG_MAX_CHUNKS_PER_FILE, 500),
    answerMaxTokens: readNumber(process.env.RAG_ANSWER_MAX_TOKENS, limits.answerMaxTokens)
  };
}

function isConfigured(config) {
  return config.provider === 'ollama' || Boolean(config.apiKey);
}

/**
 * Names the exact embedding model in use, so that changing provider or model
 * is detected and the stored vectors get rebuilt instead of silently mixing
 * two incompatible vector spaces.
 */
function embeddingModelId(config) {
  return config.provider === 'ollama' ? `ollama:${config.ollamaEmbedModel}` : `gemini:${config.embedModel}`;
}

function chatModelId(config) {
  return config.provider === 'ollama' ? config.ollamaChatModel : config.chatModel;
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/* ------------------------------------------------------------------ *
 * PDF text extraction
 * ------------------------------------------------------------------ */

let pdfjsPromise = null;
let pdfjsPaths = null;

// pdf.js insists these factory URLs end with a forward slash, so Windows
// backslash paths have to be normalised.
function toFactoryUrl(directory) {
  return `${directory.split(path.sep).join('/')}/`;
}

function getPdfjsPaths() {
  if (!pdfjsPaths) {
    const packageRoot = path.dirname(require.resolve('pdfjs-dist/package.json'));
    pdfjsPaths = {
      cmaps: toFactoryUrl(path.join(packageRoot, 'cmaps')),
      standardFontData: toFactoryUrl(path.join(packageRoot, 'standard_fonts')),
      wasm: toFactoryUrl(path.join(packageRoot, 'wasm'))
    };
  }
  return pdfjsPaths;
}

function loadPdfjs() {
  if (!pdfjsPromise) {
    pdfjsPromise = import('pdfjs-dist/legacy/build/pdf.mjs').catch(error => {
      pdfjsPromise = null;
      throw error;
    });
  }
  return pdfjsPromise;
}

function normalizeExtractedText(value) {
  return String(value || '')
    .replace(/\r\n?/g, '\n')
    .replace(/\u00a0/g, ' ')
    .replace(/[ \t]+/g, ' ')
    .replace(/ ?\n ?/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/** Releases pdf.js resources. Never throws, so teardown cannot mask a real error. */
async function teardownPdf(loadingTask, pdfDocument) {
  try {
    if (pdfDocument && typeof pdfDocument.cleanup === 'function') await pdfDocument.cleanup();
  } catch { /* ignore */ }
  try {
    if (loadingTask && typeof loadingTask.destroy === 'function') await loadingTask.destroy();
    else if (pdfDocument && typeof pdfDocument.destroy === 'function') await pdfDocument.destroy();
  } catch { /* ignore */ }
}

/**
 * Reads every page of a PDF and returns the extracted text per page.
 * Scanned PDFs hold no text layer and come back empty; the caller reports that.
 */
async function extractPdfText(filePath, { maxChars = getConfig().maxCharsPerFile } = {}) {
  const pdfjs = await loadPdfjs();
  const paths = getPdfjsPaths();
  const data = new Uint8Array(fs.readFileSync(filePath));

  const loadingTask = pdfjs.getDocument({
    data,
    cMapUrl: paths.cmaps,
    cMapPacked: true,
    standardFontDataUrl: paths.standardFontData,
    wasmUrl: paths.wasm,
    useSystemFonts: false,
    disableFontFace: true,
    isEvalSupported: false,
    useWorkerFetch: false,
    verbosity: 0
  });

  const pages = [];
  let totalChars = 0;
  let pageCount = 0;
  try {
    const pdfDocument = await loadingTask.promise;
    pageCount = pdfDocument.numPages;
    for (let pageNumber = 1; pageNumber <= pdfDocument.numPages; pageNumber += 1) {
      const page = await pdfDocument.getPage(pageNumber);
      try {
        const content = await page.getTextContent();
        let pageText = '';
        for (const item of content.items) {
          if (typeof item.str !== 'string' || !item.str) continue;
          pageText += item.str;
          if (item.hasEOL) pageText += '\n';
        }
        const cleaned = normalizeExtractedText(pageText);
        if (cleaned) {
          pages.push({ page: pageNumber, text: cleaned });
          totalChars += cleaned.length;
        }
      } finally {
        if (typeof page.cleanup === 'function') page.cleanup();
      }
      if (totalChars >= maxChars) break;
    }
  } finally {
    await teardownPdf(loadingTask, null);
  }

  return { pageCount, pages, totalChars };
}

/* ------------------------------------------------------------------ *
 * Chunking
 * ------------------------------------------------------------------ */

function collapse(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

function hardSplit(text, maxChars) {
  if (text.length <= maxChars) return [text];
  const pieces = [];
  for (let index = 0; index < text.length; index += maxChars) pieces.push(text.slice(index, index + maxChars));
  return pieces;
}

function splitIntoSentences(text) {
  const matches = text.match(/[^.!?\n]+[.!?]*\s*/g);
  return (matches || [text]).map(collapse).filter(Boolean);
}

/** Packs one page of text into overlapping chunks that fit the embedding window. */
function chunkPageText(text, { maxChars, overlap }) {
  const clean = collapse(text);
  if (!clean) return [];

  const chunks = [];
  let current = '';
  for (const sentence of splitIntoSentences(clean)) {
    for (const piece of hardSplit(sentence, maxChars)) {
      if (current && current.length + 1 + piece.length > maxChars) {
        chunks.push(current);
        const tail = overlap > 0 ? current.slice(-overlap) : '';
        current = tail ? `${tail} ${piece}` : piece;
      } else {
        current = current ? `${current} ${piece}` : piece;
      }
    }
  }
  if (current) chunks.push(current);
  return chunks.map(collapse).filter(Boolean);
}

/** Turns extracted pages into the chunk records that get embedded and stored. */
function buildChunks(pages, options = {}) {
  const config = getConfig();
  const maxChars = options.maxChars || config.chunkChars;
  const overlap = options.overlap !== undefined ? options.overlap : config.chunkOverlap;
  const maxChunks = options.maxChunks || config.maxChunksPerFile;

  const chunks = [];
  for (const page of pages) {
    for (const content of chunkPageText(page.text, { maxChars, overlap })) {
      if (chunks.length >= maxChunks) return chunks;
      chunks.push({ page: page.page, content });
    }
  }
  return chunks;
}

/* ------------------------------------------------------------------ *
 * Gemini HTTP helpers
 * ------------------------------------------------------------------ */

class AiServiceError extends Error {
  constructor(message, { status = 0, retryable = false, kind = 'request' } = {}) {
    super(message);
    this.name = 'AiServiceError';
    this.status = status;
    this.retryable = retryable;
    this.kind = kind;
  }
}

/** Turns a bare "model not found" into the exact command that fixes it. */
function missingModelError(model, error) {
  return new AiServiceError(
    `The local model "${model}" is not available yet. Pull it once with: ollama pull ${model}`,
    { status: error?.status || 0, kind: 'model' }
  );
}

function retryDelayMs(error, attempt) {
  const suggested = Number(error?.retryAfterSeconds);
  if (Number.isFinite(suggested) && suggested > 0) return Math.min(suggested * 1000, 30000);
  return Math.min(1000 * (2 ** attempt), 20000) + Math.floor(Math.random() * 250);
}

async function geminiRequest(apiPath, body, { apiKey, apiBase }) {
  let response;
  try {
    response = await fetch(`${apiBase}${apiPath}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-goog-api-key': apiKey },
      body: JSON.stringify(body)
    });
  } catch (error) {
    throw new AiServiceError(`Could not reach the AI service: ${error.message}`, { retryable: true, kind: 'network' });
  }

  const raw = await response.text();
  let payload = null;
  if (raw) {
    try {
      payload = JSON.parse(raw);
    } catch {
      payload = null;
    }
  }

  if (response.ok) return payload || {};

  const message = payload?.error?.message || raw.slice(0, 400) || `HTTP ${response.status}`;
  const status = response.status;
  const modelProblem = status === 404
    || /not found|not supported|unsupported model|does not exist|is not available/i.test(message);

  const error = new AiServiceError(message, {
    status,
    retryable: status === 429 || status === 500 || status === 503 || status === 504,
    kind: status === 401 || status === 403 ? 'auth' : modelProblem ? 'model' : status === 429 ? 'quota' : 'request'
  });

  const retryHeader = response.headers.get('retry-after');
  if (retryHeader) error.retryAfterSeconds = Number(retryHeader);
  const retryInfo = payload?.error?.details?.find(detail => String(detail['@type'] || '').includes('RetryInfo'));
  if (retryInfo?.retryDelay) error.retryAfterSeconds = parseFloat(String(retryInfo.retryDelay)) || error.retryAfterSeconds;
  throw error;
}

async function ollamaRequest(apiPath, body, { baseUrl }) {
  let response;
  try {
    response = await fetch(`${baseUrl}${apiPath}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
  } catch (error) {
    throw new AiServiceError(
      `Could not reach the local AI server at ${baseUrl}. Check that the ollama service is running. (${error.message})`,
      { retryable: true, kind: 'network' }
    );
  }

  const raw = await response.text();
  let payload = null;
  if (raw) {
    try {
      payload = JSON.parse(raw);
    } catch {
      payload = null;
    }
  }

  if (response.ok) return payload || {};

  const message = String(payload?.error || raw.slice(0, 400) || `HTTP ${response.status}`);
  const missingModel = response.status === 404 || /not found|try pulling|no such model/i.test(message);

  throw new AiServiceError(message, {
    status: response.status,
    // A missing model never fixes itself, so do not waste retries on it.
    retryable: !missingModel && response.status >= 500,
    kind: missingModel ? 'model' : 'request'
  });
}

async function withRetry(run, { attempts = 4, onRetry } = {}) {
  let lastError;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      return await run();
    } catch (error) {
      lastError = error;
      if (!(error instanceof AiServiceError) || !error.retryable || attempt === attempts - 1) throw error;
      const waitMs = retryDelayMs(error, attempt);
      if (onRetry) onRetry(error, waitMs);
      await delay(waitMs);
    }
  }
  throw lastError;
}

function toFloat32Vector(values) {
  const vector = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) vector[index] = values[index];
  return vector;
}

/** Vectors are stored unit length so cosine similarity is a plain dot product. */
function normalizeVector(vector) {
  let sum = 0;
  for (let index = 0; index < vector.length; index += 1) sum += vector[index] * vector[index];
  const magnitude = Math.sqrt(sum);
  if (!magnitude) return vector;
  for (let index = 0; index < vector.length; index += 1) vector[index] /= magnitude;
  return vector;
}

function vectorToBuffer(vector) {
  return Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength);
}

function bufferToVector(buffer) {
  const copy = new Uint8Array(buffer);
  return new Float32Array(copy.buffer, 0, Math.floor(copy.byteLength / 4));
}

function dotProduct(left, right) {
  const length = Math.min(left.length, right.length);
  let total = 0;
  for (let index = 0; index < length; index += 1) total += left[index] * right[index];
  return total;
}

/* ------------------------------------------------------------------ *
 * RAG service
 * ------------------------------------------------------------------ */

function createRag({ db, uploadDirectory, logger = console }) {
  // Which chat models are known to work on this key, newest first.
  let preferredChatModel = null;
  let indexedVectorCache = null;
  let indexingPromise = null;
  let probeCache = { at: 0, value: null };
  let timer = null;
  let stopped = true;
  const stats = { lastRunAt: null };

  db.exec(`
    CREATE TABLE IF NOT EXISTS rag_documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      source TEXT NOT NULL UNIQUE,
      document_id INTEGER,
      title TEXT NOT NULL DEFAULT '',
      document_number TEXT NOT NULL DEFAULT '',
      document_type TEXT NOT NULL DEFAULT '',
      size INTEGER NOT NULL DEFAULT 0,
      modified_ms INTEGER NOT NULL DEFAULT 0,
      page_count INTEGER NOT NULL DEFAULT 0,
      chunk_count INTEGER NOT NULL DEFAULT 0,
      content_chars INTEGER NOT NULL DEFAULT 0,
      status TEXT NOT NULL DEFAULT 'pending',
      error TEXT NOT NULL DEFAULT '',
      embedding_model TEXT NOT NULL DEFAULT '',
      indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS rag_chunks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      source TEXT NOT NULL,
      document_id INTEGER,
      page INTEGER NOT NULL DEFAULT 0,
      chunk_index INTEGER NOT NULL DEFAULT 0,
      content TEXT NOT NULL,
      embedding BLOB NOT NULL,
      embedding_model TEXT NOT NULL DEFAULT '',
      dimensions INTEGER NOT NULL DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_rag_chunks_source ON rag_chunks(source);
    CREATE INDEX IF NOT EXISTS idx_rag_chunks_document ON rag_chunks(document_id);
  `);

  // Added after the first release, so existing databases need the column put in.
  try {
    db.exec("ALTER TABLE rag_documents ADD COLUMN embedding_model TEXT NOT NULL DEFAULT ''");
  } catch (error) {
    if (!error.message.includes('duplicate column name')) throw error;
  }

  const statements = {
    getDocumentMeta: db.prepare(`
      SELECT d.id, d.caption, d.document_number, d.document_type, d.original_name,
             COALESCE(dt.requires_login, 0) AS requires_login
      FROM documents d
      LEFT JOIN document_types dt ON dt.name = d.document_type
      WHERE d.filename = ?
    `),
    getIndexedDocument: db.prepare('SELECT * FROM rag_documents WHERE source = ?'),
    listIndexedSources: db.prepare('SELECT source, size, modified_ms, status FROM rag_documents'),
    upsertDocument: db.prepare(`
      INSERT INTO rag_documents (source, document_id, title, document_number, document_type, size, modified_ms, page_count, chunk_count, content_chars, status, error, embedding_model, indexed_at)
      VALUES (@source, @document_id, @title, @document_number, @document_type, @size, @modified_ms, @page_count, @chunk_count, @content_chars, @status, @error, @embedding_model, CURRENT_TIMESTAMP)
      ON CONFLICT(source) DO UPDATE SET
        document_id = excluded.document_id,
        title = excluded.title,
        document_number = excluded.document_number,
        document_type = excluded.document_type,
        size = excluded.size,
        modified_ms = excluded.modified_ms,
        page_count = excluded.page_count,
        chunk_count = excluded.chunk_count,
        content_chars = excluded.content_chars,
        status = excluded.status,
        error = excluded.error,
        embedding_model = excluded.embedding_model,
        indexed_at = CURRENT_TIMESTAMP
    `),
    deleteChunks: db.prepare('DELETE FROM rag_chunks WHERE source = ?'),
    deleteDocument: db.prepare('DELETE FROM rag_documents WHERE source = ?'),
    insertChunk: db.prepare(`
      INSERT INTO rag_chunks (source, document_id, page, chunk_index, content, embedding, embedding_model, dimensions)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `),
    loadVectors: db.prepare('SELECT id, source, document_id, page, embedding, dimensions FROM rag_chunks'),
    fetchChunks: db.prepare('SELECT id, source, document_id, page, content FROM rag_chunks WHERE id IN (SELECT value FROM json_each(?))'),
    counts: db.prepare(`
      SELECT
        (SELECT COUNT(*) FROM rag_documents WHERE status = 'indexed' AND chunk_count > 0) AS indexed_files,
        (SELECT COUNT(*) FROM rag_documents WHERE status = 'error' OR status = 'no-text') AS problem_files,
        (SELECT COUNT(*) FROM rag_documents WHERE status = 'restricted') AS restricted_files,
        (SELECT COUNT(*) FROM rag_chunks) AS chunk_count,
        (SELECT MAX(indexed_at) FROM rag_documents WHERE status = 'indexed') AS last_indexed_at
    `)
  };

  /* ---------------- indexing ---------------- */

  function documentMetadata(source) {
    try {
      return statements.getDocumentMeta.get(source) || null;
    } catch (error) {
      if (/no such table/i.test(error.message)) return null;
      throw error;
    }
  }

  function isRestricted(metadata) {
    return Boolean(metadata && metadata.requires_login);
  }

  function removeFile(source) {
    statements.deleteChunks.run(source);
    statements.deleteDocument.run(source);
    invalidateCache();
  }

  async function indexFile(source, { force = false, knownStat = null, reason = 'scan' } = {}) {
    const filePath = path.join(uploadDirectory, source);
    let fileStat;
    try {
      fileStat = knownStat || fs.statSync(filePath);
    } catch {
      removeFile(source);
      return { source, status: 'removed' };
    }

    const metadata = documentMetadata(source);

    // Document types that need a login stay out of the index so the chatbot
    // cannot answer from restricted content.
    if (isRestricted(metadata)) {
      statements.deleteChunks.run(source);
      statements.upsertDocument.run({
        source,
        document_id: metadata.id,
        title: metadata.caption || source,
        document_number: metadata.document_number || '',
        document_type: metadata.document_type || '',
        size: fileStat.size,
        modified_ms: Math.round(fileStat.mtimeMs),
        page_count: 0,
        chunk_count: 0,
        content_chars: 0,
        status: 'restricted',
        error: 'Skipped: this document type requires a login',
        embedding_model: ''
      });
      invalidateCache();
      return { source, status: 'restricted' };
    }

    const config = getConfig();
    const modelId = embeddingModelId(config);
    const indexed = statements.getIndexedDocument.get(source);
    // A different embedding model means the stored vectors are in a different
    // vector space, so the file has to be read and embedded again.
    if (!force
      && indexed
      && indexed.status === 'indexed'
      && indexed.size === fileStat.size
      && indexed.modified_ms === Math.round(fileStat.mtimeMs)
      && indexed.embedding_model === modelId) {
      return { source, status: 'unchanged' };
    }

    const { pages, totalChars, pageCount } = await extractPdfText(filePath);
    const chunks = buildChunks(pages);
    const base = {
      source,
      document_id: metadata?.id || null,
      title: metadata?.caption || source,
      document_number: metadata?.document_number || '',
      document_type: metadata?.document_type || '',
      size: fileStat.size,
      modified_ms: Math.round(fileStat.mtimeMs),
      page_count: pageCount,
      content_chars: totalChars,
      embedding_model: modelId
    };

    if (!chunks.length) {
      statements.deleteChunks.run(source);
      statements.upsertDocument.run({ ...base, chunk_count: 0, status: 'no-text', error: 'No selectable text found. The PDF is probably a scan.' });
      invalidateCache();
      logger.warn(`[rag] ${source}: no extractable text (scanned PDF?)`);
      return { source, status: 'no-text' };
    }

    const embeddings = await embedTexts(chunks.map(chunk => chunk.content), 'RETRIEVAL_DOCUMENT');

    db.transaction(() => {
      statements.deleteChunks.run(source);
      chunks.forEach((chunk, index) => {
        const vector = embeddings[index];
        statements.insertChunk.run(
          source,
          base.document_id,
          chunk.page,
          index,
          chunk.content,
          vectorToBuffer(vector),
          modelId,
          vector.length
        );
      });
      statements.upsertDocument.run({ ...base, chunk_count: chunks.length, status: 'indexed', error: '' });
    })();
    invalidateCache();

    logger.log(`[rag] indexed ${source} (${chunks.length} chunks, ${chunks.length} embeddings, ${reason})`);
    return { source, status: 'indexed', chunks: chunks.length };
  }

  function listPdfFiles() {
    let entries;
    try {
      entries = fs.readdirSync(uploadDirectory, { withFileTypes: true });
    } catch (error) {
      if (error.code === 'ENOENT') return [];
      throw error;
    }
    return entries
      .filter(entry => entry.isFile() && /\.pdf$/i.test(entry.name) && !entry.name.startsWith('.'))
      .map(entry => entry.name);
  }

  async function runIndexPass({ force = false, reason = 'scan' } = {}) {
    const files = listPdfFiles();
    const indexedSources = new Set(statements.listIndexedSources.all().map(row => row.source));

    // Drop records for PDFs that are no longer in the folder.
    for (const source of indexedSources) {
      if (!files.includes(source)) removeFile(source);
    }

    const results = [];
    for (const source of files) {
      try {
        results.push(await indexFile(source, { force, reason }));
      } catch (error) {
        const message = error.message;
        let fileStat = null;
        try {
          fileStat = fs.statSync(path.join(uploadDirectory, source));
        } catch { /* file vanished mid-run */ }
        const metadata = documentMetadata(source);
        statements.upsertDocument.run({
          source,
          document_id: metadata?.id || null,
          title: metadata?.caption || source,
          document_number: metadata?.document_number || '',
          document_type: metadata?.document_type || '',
          size: fileStat?.size || 0,
          modified_ms: fileStat ? Math.round(fileStat.mtimeMs) : 0,
          page_count: 0,
          chunk_count: 0,
          content_chars: 0,
          status: 'error',
          error: message.slice(0, 500),
          embedding_model: ''
        });
        invalidateCache();
        logger.error(`[rag] failed to index ${source}: ${message}`);
        results.push({ source, status: 'error', error: message });

        // A missing key or an exhausted quota will fail for every file, so stop early.
        if (error instanceof AiServiceError && (error.kind === 'auth' || error.kind === 'quota')) {
          logger.error('[rag] stopping this indexing pass early because the AI service rejected the request');
          break;
        }
      }
    }

    stats.lastRunAt = new Date().toISOString();
    return results;
  }

  function invalidateCache() {
    indexedVectorCache = null;
  }

  function countsSafe() {
    try {
      return statements.counts.get() || {};
    } catch (error) {
      if (/no such table/i.test(error.message)) return {};
      throw error;
    }
  }

  /* ---------------- embeddings ---------------- */

  async function embedSlice(slice, taskType, config) {
    if (config.provider === 'ollama') {
      const prefix = taskType === 'RETRIEVAL_QUERY' ? config.ollamaQueryPrefix : config.ollamaDocPrefix;
      const inputs = slice.map(text => `${prefix}${text}`);
      try {
        const payload = await ollamaRequest('/api/embed', {
          model: config.ollamaEmbedModel,
          input: inputs,
          truncate: true,
          keep_alive: config.ollamaKeepAlive
        }, { baseUrl: config.ollamaBaseUrl });
        const embeddings = payload.embeddings || [];
        if (embeddings.length !== inputs.length) {
          throw new AiServiceError(`The local embedding model returned ${embeddings.length} vectors for ${inputs.length} passages.`);
        }
        return embeddings.map(values => normalizeVector(toFloat32Vector(values)));
      } catch (error) {
        if (error instanceof AiServiceError && error.kind === 'model') throw missingModelError(config.ollamaEmbedModel, error);
        throw error;
      }
    }

    const body = {
      requests: slice.map(text => ({
        model: `models/${config.embedModel}`,
        content: { parts: [{ text }] },
        taskType,
        outputDimensionality: config.embedDimensions
      }))
    };
    const payload = await geminiRequest(`/models/${config.embedModel}:batchEmbedContents`, body, config);
    const embeddings = payload.embeddings || [];
    if (embeddings.length !== slice.length) {
      throw new AiServiceError(`The AI service returned ${embeddings.length} embeddings for ${slice.length} passages.`);
    }
    return embeddings.map(embedding => normalizeVector(toFloat32Vector(embedding.values || [])));
  }

  async function embedTexts(texts, taskType) {
    const config = getConfig();
    if (!isConfigured(config)) throw new AiServiceError('The AI service is not configured on this server.', { kind: 'auth' });
    if (!texts.length) return [];

    const vectors = [];
    for (let offset = 0; offset < texts.length; offset += config.embedBatch) {
      const slice = texts.slice(offset, offset + config.embedBatch);
      const batch = await withRetry(
        () => embedSlice(slice, taskType, config),
        { onRetry: (error, waitMs) => logger.warn(`[rag] embedding retry in ${Math.round(waitMs)}ms: ${error.message}`) }
      );
      vectors.push(...batch);
      if (config.embedDelayMs && offset + config.embedBatch < texts.length) await delay(config.embedDelayMs);
    }
    return vectors;
  }

  /* ---------------- retrieval ---------------- */

  function loadVectorCache() {
    if (indexedVectorCache) return indexedVectorCache;
    const rows = statements.loadVectors.all();
    const entries = rows.map(row => ({
      id: row.id,
      source: row.source,
      documentId: row.document_id,
      page: row.page,
      vector: normalizeVector(bufferToVector(row.embedding))
    }));
    indexedVectorCache = entries;
    return entries;
  }

  function retrievalCandidates(questionVector, topK, minScore) {
    const entries = loadVectorCache();
    const scored = [];
    for (const entry of entries) {
      const score = dotProduct(questionVector, entry.vector);
      if (score >= minScore) scored.push({ ...entry, score });
    }
    scored.sort((left, right) => right.score - left.score);

    // Keep the best chunk per document first, then top up with nearby passages.
    const best = [];
    const seenSources = new Set();
    for (const entry of scored) {
      if (seenSources.has(entry.source)) continue;
      seenSources.add(entry.source);
      best.push(entry);
      if (best.length >= topK) return best;
    }
    for (const entry of scored) {
      if (best.includes(entry)) continue;
      best.push(entry);
      if (best.length >= topK) break;
    }
    return best;
  }

  async function retrieve(question, topK) {
    const config = getConfig();
    const [questionVector] = await embedTexts([question], 'RETRIEVAL_QUERY');
    if (!questionVector) return [];
    const matches = retrievalCandidates(questionVector, topK, config.minScore);
    if (!matches.length) return [];
    const rows = statements.fetchChunks.all(JSON.stringify(matches.map(match => match.id)));
    const byId = new Map(rows.map(row => [row.id, row]));
    return matches
      .map(match => {
        const row = byId.get(match.id);
        if (!row) return null;
        const metadata = documentMetadata(row.source);
        return {
          id: row.id,
          source: row.source,
          page: row.page,
          content: row.content,
          score: Number(match.score.toFixed(4)),
          documentId: metadata?.id || row.document_id || null,
          documentNumber: metadata?.document_number || '',
          documentType: metadata?.document_type || '',
          originalName: metadata?.original_name || row.source,
          title: metadata?.caption || row.source
        };
      })
      .filter(Boolean);
  }

  /* ---------------- answer generation ---------------- */

  function buildPrompt(question, sources) {
    const config = getConfig();
    const blocks = [];
    let used = 0;
    for (let index = 0; index < sources.length; index += 1) {
      const source = sources[index];
      const heading = [
        `[${index + 1}]`,
        source.documentNumber ? `Document number: ${source.documentNumber}` : '',
        `File: ${source.source}`,
        source.title && source.title !== source.source ? `Caption: ${source.title}` : '',
        source.page ? `Page: ${source.page}` : ''
      ].filter(Boolean).join(' | ');
      const block = `${heading}\n${source.content}`;
      if (used + block.length > config.maxContextChars) break;
      used += block.length;
      blocks.push(block);
    }
    return [
      'Answer the question using only these passages from the Rail Docs library.',
      '',
      blocks.join('\n\n---\n\n'),
      '',
      `Question: ${question}`
    ].join('\n');
  }

  function extractAnswerText(payload) {
    const candidate = payload?.candidates?.[0];
    const parts = candidate?.content?.parts || [];
    const text = parts
      .filter(part => part && typeof part.text === 'string' && !part.thought)
      .map(part => part.text)
      .join('')
      .trim();

    if (text) return text;
    const blockReason = payload?.promptFeedback?.blockReason;
    if (blockReason) return `I cannot answer that question (blocked by the AI service: ${blockReason}).`;
    if (candidate?.finishReason === 'MAX_TOKENS') {
      return 'The answer was cut off before it finished. Try asking a narrower question.';
    }
    if (candidate?.finishReason === 'SAFETY') {
      return 'I cannot answer that question because the AI service flagged the response.';
    }
    return '';
  }

  function chatModelCandidates() {
    const configured = getConfig().chatModel;
    return [...new Set([preferredChatModel, configured, ...CHAT_MODEL_FALLBACKS].filter(Boolean))];
  }

  async function generateLocalAnswer(prompt, config) {
    const request = {
      model: config.ollamaChatModel,
      messages: [
        { role: 'system', content: SYSTEM_INSTRUCTION },
        { role: 'user', content: prompt }
      ],
      stream: false,
      keep_alive: config.ollamaKeepAlive,
      options: {
        temperature: config.ollamaTemperature,
        num_ctx: config.ollamaNumCtx,
        num_predict: config.answerMaxTokens
      }
    };

    // Thinking models reason at length before answering, which is unbearable on
    // a CPU-only box. Ask them not to, and fall back to a plain request if this
    // model or Ollama version rejects the flag.
    const bodies = [{ ...request, think: false }, request];
    let lastError;
    for (const body of bodies) {
      try {
        const payload = await withRetry(
          () => ollamaRequest('/api/chat', body, { baseUrl: config.ollamaBaseUrl }),
          { attempts: 3, onRetry: (error, waitMs) => logger.warn(`[rag] answer retry in ${Math.round(waitMs)}ms: ${error.message}`) }
        );
        const text = String(payload?.message?.content || '').trim();
        if (!text) throw new AiServiceError('The local model returned an empty answer. It may have run out of its answer budget.');
        return { text, model: config.ollamaChatModel };
      } catch (error) {
        lastError = error;
        if (error instanceof AiServiceError && error.kind === 'model') throw missingModelError(config.ollamaChatModel, error);
        if (error instanceof AiServiceError && error.status === 400) continue;
        throw error;
      }
    }
    throw lastError || new AiServiceError('The local model did not return an answer.');
  }

  async function generateAnswer(prompt) {
    const config = getConfig();
    if (!isConfigured(config)) throw new AiServiceError('The AI service is not configured on this server.', { kind: 'auth' });
    if (config.provider === 'ollama') return generateLocalAnswer(prompt, config);

    const generationConfig = { maxOutputTokens: config.answerMaxTokens };
    let lastError;
    let sawModelError = false;

    for (const model of chatModelCandidates()) {
      // Gemini 3 models think by default, which is slow for a chat box. Ask for
      // low effort and drop the option if the API does not know it.
      const isGemini3 = /^gemini-3/.test(model);
      const bodies = [
        { contents: [{ role: 'user', parts: [{ text: prompt }] }], systemInstruction: { parts: [{ text: SYSTEM_INSTRUCTION }] }, generationConfig: isGemini3 ? { ...generationConfig, thinkingLevel: 'low' } : generationConfig },
        { contents: [{ role: 'user', parts: [{ text: prompt }] }], systemInstruction: { parts: [{ text: SYSTEM_INSTRUCTION }] } }
      ];

      for (const body of bodies) {
        try {
          const payload = await withRetry(
            () => geminiRequest(`/models/${model}:generateContent`, body, config),
            { attempts: 3, onRetry: (error, waitMs) => logger.warn(`[rag] answer retry in ${Math.round(waitMs)}ms: ${error.message}`) }
          );
          preferredChatModel = model;
          return { text: extractAnswerText(payload), model };
        } catch (error) {
          lastError = error;
          if (error instanceof AiServiceError && error.kind === 'auth') throw error;
          if (error instanceof AiServiceError && error.kind === 'model') {
            sawModelError = true;
            break;
          }
          // A 400 usually means an unsupported option (such as thinkingLevel on
          // an older model), so try the plain body before moving on.
          if (error instanceof AiServiceError && error.status === 400) continue;
          break;
        }
      }
    }

    if (sawModelError) {
      throw new AiServiceError(`None of the configured AI models are available on this API key (tried ${chatModelCandidates().join(', ')}). Set GEMINI_CHAT_MODEL to a model your key can use.`);
    }
    throw lastError || new AiServiceError('The AI service did not return an answer.');
  }

  /* ---------------- public API ---------------- */

  async function ask(question) {
    const config = getConfig();
    const trimmed = String(question || '').trim();
    if (!trimmed) {
      const error = new Error('Please type a question.');
      error.statusCode = 400;
      throw error;
    }
    if (trimmed.length > 1000) {
      const error = new Error('That question is too long. Please shorten it.');
      error.statusCode = 400;
      throw error;
    }

    const sources = await retrieve(trimmed, config.topK);
    if (!sources.length) {
      return {
        answer: "I could not find anything in the Rail Docs library that answers that. Try naming a document number, or use fewer and more specific words.",
        sources: [],
        model: null
      };
    }

    const { text, model } = await generateAnswer(buildPrompt(trimmed, sources));
    const answer = text || 'The AI service returned an empty answer. Please try rephrasing the question.';
    return {
      answer,
      model,
      sources: sources.map(source => ({
        documentId: source.documentId,
        source: source.source,
        originalName: source.originalName,
        documentNumber: source.documentNumber,
        documentType: source.documentType,
        title: source.title,
        page: source.page,
        score: source.score
      }))
    };
  }

  /**
   * Reports whether the chosen provider is actually reachable and has its
   * models. Cached briefly so the chat UI can poll it cheaply.
   */
  async function probeProvider() {
    const config = getConfig();
    if (probeCache.value && Date.now() - probeCache.at < 15000) return probeCache.value;

    let value;
    if (config.provider === 'ollama') {
      const wanted = [config.ollamaEmbedModel, config.ollamaChatModel];
      try {
        const response = await fetch(`${config.ollamaBaseUrl}/api/tags`, { signal: AbortSignal.timeout(5000) });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        const names = (data.models || []).map(entry => String(entry.name || entry.model || '')).filter(Boolean);
        const missing = wanted.filter(need => !names.some(name => name === need || name === `${need}:latest`));
        value = { online: true, models: names, missing, hint: '' };
      } catch (error) {
        value = {
          online: false,
          models: [],
          missing: wanted,
          hint: `The local AI server at ${config.ollamaBaseUrl} is not responding. Check that the ollama service is running.`
        };
      }
    } else {
      value = {
        online: Boolean(config.apiKey),
        models: [config.chatModel, config.embedModel],
        missing: [],
        hint: config.apiKey ? '' : 'GEMINI_API_KEY is not set on the server.'
      };
    }

    probeCache = { at: Date.now(), value };
    return value;
  }

  async function getStatus() {
    const config = getConfig();
    const counts = countsSafe();
    const probe = await probeProvider();
    return {
      enabled: isConfigured(config),
      provider: config.provider,
      online: probe.online,
      missingModels: probe.missing,
      hint: probe.hint,
      indexing: Boolean(indexingPromise),
      // Only documents that actually produced passages can be answered from.
      indexedFiles: counts.indexed_files || 0,
      problemFiles: counts.problem_files || 0,
      restrictedFiles: counts.restricted_files || 0,
      chunks: counts.chunk_count || 0,
      lastIndexedAt: counts.last_indexed_at || null,
      lastRunAt: stats.lastRunAt,
      chatModel: config.provider === 'ollama' ? config.ollamaChatModel : (preferredChatModel || config.chatModel),
      embedModel: config.provider === 'ollama' ? config.ollamaEmbedModel : config.embedModel
    };
  }

  function isIndexing() {
    return Boolean(indexingPromise);
  }

  async function indexNow({ force = false, reason = 'manual' } = {}) {
    if (indexingPromise) return indexingPromise;
    indexingPromise = runIndexPass({ force, reason })
      .catch(error => {
        logger.error(`[rag] indexing pass failed: ${error.message}`);
        throw error;
      })
      .finally(() => {
        indexingPromise = null;
      });
    return indexingPromise;
  }

  function start() {
    if (timer) return;
    stopped = false;
    const config = getConfig();
    if (!isConfigured(config)) {
      logger.warn('[rag] no AI provider is configured, so the library assistant stays off. Set AI_PROVIDER=ollama and OLLAMA_BASE_URL, or set GEMINI_API_KEY. The rest of the site works normally.');
      return;
    }

    // Surface provider problems at start up rather than on the first question.
    probeProvider()
      .then(probe => {
        if (!probe.online) logger.warn(`[rag] ${probe.hint || 'the AI provider is not reachable yet.'}`);
        else if (probe.missing.length) logger.warn(`[rag] local models are not downloaded yet: ${probe.missing.join(', ')}. Pull them with: ollama pull ${probe.missing.join(' ')}`);
      })
      .catch(() => {});

    const tick = async () => {
      if (stopped) return;
      try {
        if (!indexingPromise) await indexNow({ reason: 'scan' });
      } catch (error) {
        logger.error(`[rag] background indexing failed: ${error.message}`);
      }
      if (stopped) return;
      timer = setTimeout(tick, Math.max(config.rescanIntervalMs, 15000));
      if (timer.unref) timer.unref();
    };
    timer = setTimeout(tick, 2000);
    if (timer.unref) timer.unref();
  }

  function stop() {
    stopped = true;
    if (timer) clearTimeout(timer);
    timer = null;
  }

  return {
    ask,
    getStatus,
    isIndexing,
    indexNow,
    // Force only when a re-embed is genuinely wanted: a file with no index
    // record (or a changed one) is picked up without it, and restricted types
    // are re-checked before the skip logic runs.
    indexFile: (source, options = {}) => indexFile(source, { force: false, reason: 'upload', ...options }),
    removeFile,
    start,
    stop,
    isEnabled: () => isConfigured(getConfig()),
    invalidateCache
  };
}

module.exports = {
  createRag,
  extractPdfText,
  buildChunks,
  chunkPageText,
  normalizeExtractedText,
  AiServiceError,
  getConfig
};
