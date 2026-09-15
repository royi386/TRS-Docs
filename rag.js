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
const os = require('os');
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
const LOCAL_LIMITS = { topK: 4, maxContextChars: 4000, answerMaxTokens: 400, embedBatch: 16, embedDelayMs: 0, historyTurns: 2, historyChars: 1200 };
const CLOUD_LIMITS = { topK: 6, maxContextChars: 12000, answerMaxTokens: 2048, embedBatch: 24, embedDelayMs: 250, historyTurns: 4, historyChars: 2400 };

// OCR (scanned PDFs). Tesseract runs as WebAssembly inside the Node process, so
// no system packages are needed. Language data downloads once from
// tessdata.projectnaptha.com and is cached in a local folder; point
// RAG_OCR_LANG_PATH at a folder holding <lang>.traineddata.gz for fully
// offline servers.
const DEFAULT_OCR_LANGS = 'eng';
const OCR_CACHE_DIR = path.join(__dirname, '.ocr-cache');
const OCR_PAGE_TARGET_WIDTH = 1700;
// A cap on the rendered page width keeps big scanned drawings from creating a
// bitmap large enough to exhaust the process memory.
const PDF_RENDER_MAX_WIDTH = 4000;
const OCR_MAX_PAGES = 40;

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
  '8. When you mention a document, use its document number and caption from the passage header.',
  '9. The conversation history shows what was discussed earlier. Use it to understand short follow-up questions, but answer only from the passages, never from memory of the earlier conversation.'
].join('\n');

/**
 * Trims the stored conversation to the last few turns within a character
 * budget, so a long session cannot inflate the prompt on the small local
 * models. Returns '' when there is nothing to include.
 */
function historyBlock(history, config) {
  if (!Array.isArray(history) || !history.length) return '';
  const wanted = [];
  let used = 0;
  for (let index = history.length - 1; index >= 0; index -= 1) {
    const turn = history[index];
    const question = String(turn?.question || '').slice(0, 500).trim();
    const answer = String(turn?.answer || '').slice(0, 800).trim();
    if (!question) continue;
    const line = `Q: ${question}\nA: ${answer || '(no answer)'}`;
    if (used + line.length > config.historyChars && wanted.length) break;
    used += line.length;
    wanted.unshift(line);
    if (wanted.length >= config.historyTurns) break;
  }
  return wanted.join('\n\n');
}

/**
 * Follow-up questions such as "and its torque?" are meaningless on their own,
 * so retrieval searches for a rewritten, self-contained question instead. The
 * rewrite is heuristic and free: last question + last answer give the pronouns
 * their antecedents without an extra model round-trip.
 */
function standaloneQuestion(question, history) {
  const previous = Array.isArray(history) ? history[history.length - 1] : null;
  const priorQuestion = String(previous?.question || '').trim();
  const priorAnswer = String(previous?.answer || '').trim();
  if (!priorQuestion || priorQuestion === question) return question;

  // Only rewrite when the question actually leans on the previous turn: it
  // opens with a dangling reference, mentions one mid-sentence, or is a short
  // fragment.
  const needsContext = /^(and|also|its|it's|it\b|they|their|them|that\b|this\b|those|these|same|so\b|then|what about|how about)\b/i.test(question)
    || (/\b(it|its|itself|their|them|same|above|earlier|aforementioned)\b/i.test(question) && question.length < 150)
    || question.length < 25;
  if (!needsContext) return question;

  // A short answer usually names the thing the follow-up is about (a document
  // number, a component...), so lead with it.
  const subject = priorAnswer
    .replace(/\[\d+\]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .split(/[.!?\n]/)[0]
    .slice(0, 160)
    .trim();
  const context = subject && subject.length > 15 ? subject : priorQuestion;
  return `${context} — ${question}`;
}

/**
 * Builds an FTS5 MATCH expression from a question. Terms are OR-ed so a passage
 * mentioning most of the asked things outranks one mentioning a single word;
 * stopwords and short tokens are dropped, and numbers (document numbers, values
 * like 0003 or 260) are kept verbatim because they are the strongest signal.
 * Returns '' when the question has nothing worth a keyword lookup.
 */
function keywordQueryFromQuestion(question) {
  const stopWords = new Set(['the', 'a', 'an', 'of', 'in', 'on', 'for', 'to', 'is', 'are', 'was', 'were', 'be', 'been', 'and', 'or', 'what', 'which', 'who', 'whom', 'how', 'why', 'when', 'where', 'does', 'do', 'did', 'can', 'could', 'should', 'would', 'will', 'shall', 'may', 'might', 'must', 'give', 'tell', 'show', 'list', 'find', 'any', 'its', 'it', 'his', 'her', 'their', 'there', 'that', 'this', 'these', 'those', 'from', 'with', 'as', 'at', 'by', 'per', 'not', 'no', 'yes', 'about', 'into', 'over', 'under', 'please']);
  const terms = [];
  const seen = new Set();
  for (const raw of String(question || '').toLowerCase().split(/[^a-z0-9]+/)) {
    const term = raw.trim();
    if (!term || term.length < 2 || seen.has(term)) continue;
    if (!/\d/.test(term) && (stopWords.has(term) || term.length < 3)) continue;
    seen.add(term);
    terms.push(term);
    if (terms.length >= 12) break;
  }
  if (!terms.length) return '';
  // Double quotes are escaped by doubling; FTS5 treats a quoted token as a
  // plain string, so punctuation-free terms never form operators.
  return terms.map(term => `"${term}"`).join(' OR ');
}

function readNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function getConfig() {
  const requested = String(process.env.AI_PROVIDER || '').trim().toLowerCase();
  const provider = requested === 'gemini' || requested === 'ollama' ? requested : 'ollama';
  const limits = provider === 'ollama' ? LOCAL_LIMITS : CLOUD_LIMITS;
  const topK = Math.min(readNumber(process.env.RAG_TOP_K, limits.topK), 20);

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
    topK,
    maxContextChars: readNumber(process.env.RAG_MAX_CONTEXT_CHARS, limits.maxContextChars),
    minScore: Number.isFinite(Number(process.env.RAG_MIN_SCORE)) ? Number(process.env.RAG_MIN_SCORE) : 0.3,
    chunkChars: readNumber(process.env.RAG_CHUNK_CHARS, 1400),
    chunkOverlap: readNumber(process.env.RAG_CHUNK_OVERLAP, 200),
    embedBatch: Math.min(readNumber(process.env.RAG_EMBED_BATCH, limits.embedBatch), EMBED_BATCH_MAX),
    embedDelayMs: Number.isFinite(Number(process.env.RAG_EMBED_DELAY_MS)) ? Math.max(Number(process.env.RAG_EMBED_DELAY_MS), 0) : limits.embedDelayMs,
    rescanIntervalMs: readNumber(process.env.RAG_RESCAN_INTERVAL_MS, 5 * 60 * 1000),
    maxCharsPerFile: readNumber(process.env.RAG_MAX_CHARS_PER_FILE, 1500000),
    maxChunksPerFile: readNumber(process.env.RAG_MAX_CHUNKS_PER_FILE, 500),
    answerMaxTokens: readNumber(process.env.RAG_ANSWER_MAX_TOKENS, limits.answerMaxTokens),
    historyTurns: Math.min(readNumber(process.env.RAG_HISTORY_TURNS, limits.historyTurns), 10),
    historyChars: readNumber(process.env.RAG_HISTORY_CHARS, limits.historyChars),
    ocrEnabled: process.env.RAG_OCR !== '0',
    ocrLangs: (process.env.RAG_OCR_LANGS || DEFAULT_OCR_LANGS).split(',').map(part => part.trim()).filter(Boolean),
    ocrLangPath: process.env.RAG_OCR_LANG_PATH || '',
    ocrCacheDir: process.env.RAG_OCR_CACHE_DIR || OCR_CACHE_DIR,
    ocrMaxPages: Math.min(readNumber(process.env.RAG_OCR_MAX_PAGES, OCR_MAX_PAGES), 200),
    ocrDpiScale: Number.isFinite(Number(process.env.RAG_OCR_DPI_SCALE)) ? Math.min(Math.max(Number(process.env.RAG_OCR_DPI_SCALE), 1), 3) : 2,
    // Hybrid retrieval. The keyword half uses SQLite FTS5 (bm25). Weights are a
    // blend: vectors carry meaning, keywords carry exact terms like document
    // numbers and named limits, and bm25 rewards short passages that mention
    // what was asked several times.
    keywordWeight: Number.isFinite(Number(process.env.RAG_KEYWORD_WEIGHT)) ? Math.max(Number(process.env.RAG_KEYWORD_WEIGHT), 0) : 0.5,
    vectorWeight: Number.isFinite(Number(process.env.RAG_VECTOR_WEIGHT)) ? Math.max(Number(process.env.RAG_VECTOR_WEIGHT), 0) : 0.5,
    perDocumentLimit: Math.max(1, Math.min(readNumber(process.env.RAG_PER_DOC_LIMIT, 3), topK)),
    retrievePool: Math.max(topK, Math.min(readNumber(process.env.RAG_RETRIEVE_POOL, topK * 4), 40))
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

// Canvas for pdf.js to draw on. @napi-rs/canvas ships prebuilt binaries, so it
// works in the Docker image without any system packages. pdf.js also uses this
// factory for its own scratch canvases (soft masks, patterns) while rendering.
let napiCanvasPromise = null;

function loadNapiCanvas() {
  if (!napiCanvasPromise) {
    napiCanvasPromise = import('@napi-rs/canvas').catch(error => {
      napiCanvasPromise = null;
      throw error;
    });
  }
  return napiCanvasPromise;
}

async function createCanvasFactory() {
  const canvasModule = await loadNapiCanvas();
  return {
    create(width, height) {
      const canvas = canvasModule.createCanvas(Math.max(1, width), Math.max(1, height));
      return { canvas, context: canvas.getContext('2d') };
    },
    reset(canvasAndContext, width, height) {
      canvasAndContext.canvas.width = Math.max(1, width);
      canvasAndContext.canvas.height = Math.max(1, height);
    },
    destroy(canvasAndContext) {
      if (canvasAndContext?.canvas && typeof canvasAndContext.canvas.dispose === 'function') canvasAndContext.canvas.dispose();
      if (canvasAndContext) {
        canvasAndContext.canvas = null;
        canvasAndContext.context = null;
      }
    }
  };
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
 * Normalise any accepted source (path, Buffer, Uint8Array) into a plain
 * Uint8Array. pdf.js takes ownership of the ArrayBuffer it is handed and can
 * detach it, so it always gets its own copy and the caller's bytes stay valid
 * for reuse by the next pass.
 */
function toPdfBytes(source) {
  const buffer = Buffer.isBuffer(source) ? source
    : source instanceof Uint8Array ? Buffer.from(source.buffer, source.byteOffset, source.byteLength)
    : fs.readFileSync(source);
  return new Uint8Array(buffer);
}

/**
 * Opens a PDF for text extraction or rendering. Passing already-read bytes
 * avoids a second read of the file, which matters for the large PDFs this
 * library accepts: big files are read once and used for both passes.
 */
async function openPdfDocument(source) {
  const pdfjs = await loadPdfjs();
  const paths = getPdfjsPaths();
  const canvasFactory = await createCanvasFactory();
  const data = toPdfBytes(source);

  return pdfjs.getDocument({
    data,
    canvasFactory,
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
}

/**
 * Reads every page of a PDF and returns the extracted text per page.
 * Scanned PDFs hold no text layer and come back empty; the caller reports that.
 * A page that cannot be read is skipped, so one bad page no longer hides the
 * text of every page after it.
 */
async function extractPdfText(filePath, { maxChars = getConfig().maxCharsPerFile, bytes = null } = {}) {
  const loadingTask = await openPdfDocument(bytes || filePath);

  const pages = [];
  let totalChars = 0;
  let pageCount = 0;
  try {
    const pdfDocument = await loadingTask.promise;
    pageCount = pdfDocument.numPages;
    for (let pageNumber = 1; pageNumber <= pdfDocument.numPages; pageNumber += 1) {
      try {
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
      } catch (error) {
        console.warn(`[rag] page ${pageNumber} of ${filePath} could not be read: ${error.message}`);
      }
      if (totalChars >= maxChars) break;
    }
  } finally {
    await teardownPdf(loadingTask, null);
  }

  return { pageCount, pages, totalChars };
}

/* ------------------------------------------------------------------ *
 * OCR for scanned PDFs
 * ------------------------------------------------------------------ */

// tesseract.js is only loaded when a scanned PDF shows up, so normal
// deployments never pay its start up cost.
let tesseractPromise = null;

function loadTesseract() {
  if (!tesseractPromise) {
    tesseractPromise = import('tesseract.js').catch(error => {
      tesseractPromise = null;
      throw error;
    });
  }
  return tesseractPromise;
}

// One Tesseract worker per core through a scheduler, so indexing a batch of
// scans uses the whole CPU instead of a single core. Workers are pooled and
// stay warm between files: initialising one costs seconds, which would
// otherwise be paid for every PDF.
let ocrSchedulerPromise = null;
let ocrSchedulerKey = '';

function ocrWorkerCount(pageCount) {
  const configured = readNumber(process.env.RAG_OCR_WORKERS, 0);
  const cores = Math.max(1, os.cpus().length - 1); // leave one core for the app itself
  return Math.max(1, Math.min(configured || cores, cores, pageCount || 1, 8));
}

async function getOcrScheduler(config, pageCount) {
  const langs = config.ocrLangs.join('+') || DEFAULT_OCR_LANGS;
  const key = `${langs}|${config.ocrLangPath}|${config.ocrCacheDir}`;
  if (ocrSchedulerPromise && ocrSchedulerKey !== key) {
    const stale = ocrSchedulerPromise;
    ocrSchedulerPromise = null;
    try {
      const { scheduler } = await stale;
      await scheduler.terminate();
    } catch { /* ignore */ }
  }
  if (!ocrSchedulerPromise) {
    ocrSchedulerKey = key;
    ocrSchedulerPromise = (async () => {
      const { createScheduler, createWorker } = await loadTesseract();
      const scheduler = createScheduler();
      const options = {
        cachePath: config.ocrCacheDir,
        logger: () => {} // keep recognition progress out of the Docker logs
      };
      if (config.ocrLangPath) options.langPath = config.ocrLangPath;
      fs.mkdirSync(config.ocrCacheDir, { recursive: true });
      const workerCount = ocrWorkerCount(pageCount);
      await Promise.all(Array.from({ length: workerCount }, () =>
        createWorker(langs, 1, options).then(worker => scheduler.addWorker(worker))));
      return { scheduler, workerCount };
    })();
    ocrSchedulerPromise.catch(() => {
      if (ocrSchedulerKey === key) ocrSchedulerPromise = null;
    });
  }
  return ocrSchedulerPromise;
}

async function stopOcrScheduler() {
  if (!ocrSchedulerPromise) return;
  const pending = ocrSchedulerPromise;
  ocrSchedulerPromise = null;
  try {
    const { scheduler } = await pending;
    await scheduler.terminate();
  } catch { /* ignore */ }
}

function withOcrWorkers(config, pageCount, fn) {
  return getOcrScheduler(config, pageCount).then(({ scheduler }) =>
    // recognise(image, options, output, jobId) - the explicit defaults keep the
    // scheduler's internal job id from landing in the options slot.
    fn(image => scheduler.addJob('recognize', image, {}, { text: true })));
}

/** Renders one PDF page into an RGB canvas sized for OCR. */
async function renderPdfPage(pdfDocument, pageNumber, scale) {
  const canvasModule = await loadNapiCanvas();
  const page = await pdfDocument.getPage(pageNumber);
  try {
    const full = page.getViewport({ scale });
    // Wide drawings are scaled down to stay inside the bitmap budget; Tesseract
    // still reads the capped size well.
    const scaleDown = Math.min(1, PDF_RENDER_MAX_WIDTH / Math.max(1, full.width));
    const viewport = scaleDown < 1 ? page.getViewport({ scale: scale * scaleDown }) : full;
    const canvas = canvasModule.createCanvas(Math.max(1, Math.floor(viewport.width)), Math.max(1, Math.floor(viewport.height)));
    const context = canvas.getContext('2d');
    context.fillStyle = 'white';
    context.fillRect(0, 0, canvas.width, canvas.height);
    await page.render({ canvasContext: context, viewport, background: 'white' }).promise;
    return canvas;
  } finally {
    if (typeof page.cleanup === 'function') page.cleanup();
  }
}

/**
 * Rasterises pages with pdf.js and reads them back with Tesseract. Returns the
 * same shape as extractPdfText, plus `ocrPages`: the page numbers that were
 * read. `onlyPages` restricts the run to the given page numbers, which is how
 * the scanned pages inside a mostly text PDF are read without repeating the
 * rest. A page that fails is skipped, not fatal.
 */
async function ocrPdfPages(filePath, { maxChars = getConfig().maxCharsPerFile, bytes = null, onlyPages = null } = {}) {
  const config = getConfig();
  if (!config.ocrEnabled) throw new Error('OCR is disabled (RAG_OCR=0)');

  const loadingTask = await openPdfDocument(bytes || filePath);

  const pages = [];
  let totalChars = 0;
  let pageCount = 0;
  let pdfDocument = null;
  try {
    pdfDocument = await loadingTask.promise;
    pageCount = pdfDocument.numPages;

    let wanted = onlyPages && onlyPages.length
      ? [...new Set(onlyPages)].filter(pageNumber => pageNumber >= 1 && pageNumber <= pdfDocument.numPages)
      : null;
    const pageLimit = Math.min(wanted ? wanted.length : pdfDocument.numPages, config.ocrMaxPages);
    if (!wanted && pdfDocument.numPages > pageLimit) {
      console.warn(`[rag] OCR: only the first ${pageLimit} of ${pdfDocument.numPages} pages will be read`);
    }
    wanted = wanted ? wanted.slice(0, pageLimit) : Array.from({ length: pageLimit }, (_, index) => index + 1);

    pages.push(...await withOcrWorkers(config, wanted.length, async recognize => {
      // Pages are rendered on this thread and recognised on the worker pool, so
      // several pages are in flight at once and all cores stay busy.
      const jobs = wanted.map(pageNumber => (async () => {
        try {
          const canvas = await renderPdfPage(pdfDocument, pageNumber, config.ocrDpiScale);
          let image;
          try {
            // tesseract.js in Node wants a Buffer, not a canvas object.
            image = canvas.toBuffer('image/png');
          } finally {
            // @napi-rs/canvas exposes a native bitmap, so free it promptly.
            if (typeof canvas?.dispose === 'function') canvas.dispose();
          }
          const { data: { text } } = await recognize(image);
          const cleaned = normalizeExtractedText(text);
          if (cleaned) {
            totalChars += cleaned.length;
            return { page: pageNumber, text: cleaned };
          }
          return null;
        } catch (error) {
          console.warn(`[rag] OCR: page ${pageNumber} of ${filePath} failed: ${error.message}`);
          return null;
        }
      })());
      const settled = await Promise.all(jobs);
      return settled.filter(Boolean);
    }));
  } finally {
    await teardownPdf(loadingTask, pdfDocument);
  }

  return { pageCount, pages, totalChars, ocr: true, mixed: Boolean(onlyPages && onlyPages.length), ocrPages: pages.map(page => page.page) };
}

/**
 * OCR pass for PDFs that already carry some native text: only the pages with
 * no text layer are rasterised and read, so scanned pages inside a mostly text
 * PDF are recovered instead of silently lost. Null when nothing is missing.
 */
async function ocrMissingTextPages(filePath, extracted, options = {}) {
  if (!getConfig().ocrEnabled) return null;
  const haveText = new Set(extracted.pages.map(page => page.page));
  const missing = [];
  for (let pageNumber = 1; pageNumber <= extracted.pageCount; pageNumber += 1) {
    if (!haveText.has(pageNumber)) missing.push(pageNumber);
  }
  if (!missing.length) return null;
  return ocrPdfPages(filePath, { ...options, onlyPages: missing });
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

    CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(
      content,
      tokenize = 'porter unicode61'
      , content='rag_chunks', content_rowid='id'
    );
  `);

  // External-content FTS keeps a copy of nothing but the index; it must be
  // filled once for tables that predate it and kept in sync by the triggers
  // from then on. Rebuilding on every boot would re-tokenise the whole
  // library, so only rebuild when chunks exist but the index is empty.
  try {
    const chunkCount = db.prepare('SELECT COUNT(*) AS n FROM rag_chunks').get().n;
    const ftsCount = db.prepare('SELECT COUNT(*) AS n FROM rag_chunks_fts').get().n;
    if (chunkCount > 0 && ftsCount === 0) {
      db.exec("INSERT INTO rag_chunks_fts(rag_chunks_fts) VALUES ('rebuild')");
    }
  } catch (error) {
    if (!/no such table/i.test(error.message)) throw error;
  }
  try {
    db.exec(`
      CREATE TRIGGER IF NOT EXISTS rag_chunks_ai AFTER INSERT ON rag_chunks BEGIN
        INSERT INTO rag_chunks_fts(rowid, content) VALUES (new.id, new.content);
      END;
      CREATE TRIGGER IF NOT EXISTS rag_chunks_ad AFTER DELETE ON rag_chunks BEGIN
        INSERT INTO rag_chunks_fts(rag_chunks_fts, rowid, content) VALUES ('delete', old.id, old.content);
      END;
    `);
  } catch (error) {
    if (!/no such table/i.test(error.message)) throw error;
  }

  // Added after the first release, so existing databases need the column put in.
  try {
    db.exec("ALTER TABLE rag_documents ADD COLUMN embedding_model TEXT NOT NULL DEFAULT ''");
  } catch (error) {
    if (!error.message.includes('duplicate column name')) throw error;
  }
  // How many pages were recovered by OCR (0 = the PDF had a native text layer).
  try {
    db.exec('ALTER TABLE rag_documents ADD COLUMN ocr_pages INTEGER NOT NULL DEFAULT 0');
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
    // Problem files first, then OCR-rescued scans, then the rest alphabetically.
    listFiles: db.prepare(`
      SELECT source, document_number, title, page_count, chunk_count, status, ocr_pages AS ocrPages, error, indexed_at AS indexedAt
      FROM rag_documents
      ORDER BY status <> 'indexed', ocr_pages > 0 DESC, source COLLATE NOCASE
    `),
    upsertDocument: db.prepare(`
      INSERT INTO rag_documents (source, document_id, title, document_number, document_type, size, modified_ms, page_count, chunk_count, content_chars, status, error, embedding_model, ocr_pages, indexed_at)
      VALUES (@source, @document_id, @title, @document_number, @document_type, @size, @modified_ms, @page_count, @chunk_count, @content_chars, @status, @error, @embedding_model, @ocr_pages, CURRENT_TIMESTAMP)
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
        ocr_pages = excluded.ocr_pages,
        indexed_at = CURRENT_TIMESTAMP
    `),
    deleteChunks: db.prepare('DELETE FROM rag_chunks WHERE source = ?'),
    deleteDocument: db.prepare('DELETE FROM rag_documents WHERE source = ?'),
    insertChunk: db.prepare(`
      INSERT INTO rag_chunks (source, document_id, page, chunk_index, content, embedding, embedding_model, dimensions)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `),
    loadVectors: db.prepare('SELECT id, source, document_id, page, embedding, dimensions FROM rag_chunks'),
    // bm25() is negative (more relevant = more negative), so it sorts ascending.
    searchKeywords: db.prepare(`
      SELECT rowid AS id, bm25(rag_chunks_fts, 8.0) AS rank
      FROM rag_chunks_fts
      WHERE rag_chunks_fts MATCH ?
      ORDER BY rank
      LIMIT ?
    `),
    fetchChunks: db.prepare('SELECT id, source, document_id, page, content FROM rag_chunks WHERE id IN (SELECT value FROM json_each(?))'),
    counts: db.prepare(`
      SELECT
        (SELECT COUNT(*) FROM rag_documents WHERE status = 'indexed' AND chunk_count > 0) AS indexed_files,
        (SELECT COUNT(*) FROM rag_documents WHERE status = 'error' OR status = 'no-text') AS problem_files,
        (SELECT COUNT(*) FROM rag_documents WHERE status = 'restricted') AS restricted_files,
        (SELECT COUNT(*) FROM rag_documents WHERE ocr_pages > 0) AS ocr_files,
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
        embedding_model: '',
        ocr_pages: 0
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

    // Large files are read once: the same bytes feed both the text layer pass
    // and, when needed, the OCR pass.
    const bytes = fs.readFileSync(filePath);

    const { pages: nativePages, totalChars: nativeChars, pageCount } = await extractPdfText(filePath, { bytes });
    let pages = nativePages;
    let totalChars = nativeChars;

    // Pages with no text layer are read with OCR: whole scanned PDFs have every
    // page missing, while hybrid PDFs (text pages mixed with scanned pages, the
    // norm in big drawings) only need their scanned pages recovered.
    let ocrError = '';
    let ocrPages = 0;
    if (getConfig().ocrEnabled) {
      try {
        const ocr = totalChars
          ? await ocrMissingTextPages(filePath, { pages, pageCount }, { bytes })
          : await ocrPdfPages(filePath, { bytes });
        if (ocr && ocr.pages.length) {
          pages = totalChars
            ? [...pages, ...ocr.pages].sort((left, right) => left.page - right.page)
            : ocr.pages;
          totalChars += ocr.totalChars;
          ocrPages = ocr.pages.length;
          logger.log(`[rag] ${source}: OCR recovered ${ocrPages} page(s) of text`);
        }
      } catch (error) {
        ocrError = error.message;
        logger.warn(`[rag] ${source}: OCR failed: ${error.message}`);
      }
    }

    let chunks = buildChunks(pages);

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
      embedding_model: modelId,
      ocr_pages: ocrPages
    };

    if (!chunks.length) {
      statements.deleteChunks.run(source);
      const reason = ocrError
        ? `Scanned PDF, and OCR could not read it: ${ocrError}`
        : 'No readable text found, even after OCR. The scan quality is probably too low.';
      statements.upsertDocument.run({ ...base, chunk_count: 0, status: 'no-text', error: reason });
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

  function retrievalCandidates(questionVector, topK, minScore, keywordQuery = '') {
    const config = getConfig();
    const entries = loadVectorCache();

    // Vector pass: semantic similarity, normalised to 0..1 (cosine of unit
    // vectors runs roughly 0.3..0.9 for useful matches, so scale and clamp).
    const vectorScores = new Map();
    for (const entry of entries) {
      const score = dotProduct(questionVector, entry.vector);
      if (score >= minScore) vectorScores.set(entry.id, Math.min(1, Math.max(0, (score - minScore) / (1 - minScore || 1))));
    }

    // Keyword pass: FTS5 bm25 finds the exact terms — document numbers, named
    // quantities, part names — that embeddings routinely rank below merely
    // similar-looking text. bm25 is negative-by-relevance; scores are normalised
    // relative to the best hit (1.0) so they are comparable with vector scores
    // regardless of how large the raw bm25 magnitudes are.
    const keywordScores = new Map();
    if (keywordQuery) {
      try {
        const rows = statements.searchKeywords.all(keywordQuery, config.retrievePool);
        const best = rows.length ? rows[0].rank : 0; // most negative = most relevant
        for (const row of rows) {
          keywordScores.set(row.id, best < 0 ? Math.min(1, row.rank / best) : 1);
        }
      } catch (error) {
        logger.warn(`[rag] keyword search skipped: ${error.message}`);
      }
    }

    // Fuse: weighted sum of the two normalised signals. A passage strong in
    // both (the real answer) beats a passage that merely sounds similar.
    const fused = new Map();
    for (const [id, score] of vectorScores) fused.set(id, score * config.vectorWeight);
    for (const [id, score] of keywordScores) fused.set(id, (fused.get(id) || 0) + score * config.keywordWeight);

    const scored = [...fused.entries()]
      .map(([id, score]) => {
        const entry = entries.find(candidate => candidate.id === id);
        return entry ? { ...entry, score } : null;
      })
      .filter(Boolean)
      .sort((left, right) => right.score - left.score);

    // Strongest passages first, but never more than perDocumentLimit passages
    // from one document. Unlike the old forced diversity, the 2nd-best chunk of
    // the document that actually holds the answer still makes the cut — the
    // model then cites a page that really contains what was asked.
    const best = [];
    const perSource = new Map();
    for (const entry of scored) {
      const used = perSource.get(entry.source) || 0;
      if (used >= config.perDocumentLimit) continue;
      perSource.set(entry.source, used + 1);
      best.push(entry);
      if (best.length >= topK) return best;
    }
    return best;
  }

  async function retrieve(question, topK) {
    const config = getConfig();
    const [questionVector] = await embedTexts([question], 'RETRIEVAL_QUERY');
    if (!questionVector) return [];
    const keywordQuery = keywordQueryFromQuestion(question);
    const matches = retrievalCandidates(questionVector, topK, config.minScore, keywordQuery);
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

  function buildPrompt(question, sources, history = '') {
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
      history ? `Earlier in this conversation:\n${history}\n` : '',
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

  async function ask(question, { history = [] } = {}) {
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

    // Retrieval runs on a self-contained version of the question so pronouns
    // from a follow-up still find the right passages.
    const searchQuestion = standaloneQuestion(trimmed, history);
    if (searchQuestion !== trimmed) {
      logger.log(`[rag] follow-up detected, searching for: ${searchQuestion}`);
    }
    const sources = await retrieve(searchQuestion, config.topK);
    if (!sources.length) {
      return {
        answer: "I could not find anything in the Rail Docs library that answers that. Try naming a document number, or use fewer and more specific words.",
        sources: [],
        model: null
      };
    }

    const { text, model } = await generateAnswer(buildPrompt(trimmed, sources, historyBlock(history, config)));
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
      ocrFiles: counts.ocr_files || 0,
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
    stopOcrScheduler();
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
    listFiles: () => statements.listFiles.all(),
    invalidateCache
  };
}

module.exports = {
  createRag,
  extractPdfText,
  ocrPdfPages,
  ocrMissingTextPages,
  buildChunks,
  chunkPageText,
  normalizeExtractedText,
  historyBlock,
  standaloneQuestion,
  keywordQueryFromQuestion,
  AiServiceError,
  getConfig
};
