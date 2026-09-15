# Rail Docs

Searchable Indian Railways RDSO SMI, MS and TC PDF library. Engineers can search by document number or caption and download matching PDFs. The `Admin` page requires the server-side `ADMIN_PASSWORD` and is the only way to upload documents.

## Run with Docker on OMV

1. Copy this folder to the OMV server at `10.189.34.56`.
2. Edit `docker-compose.yml` and replace `change-this-to-a-long-password`.
3. Start it with `docker compose up -d --build`.
4. Open `http://10.189.34.56:3000` on the railway network.

The `data` folder stores SQLite metadata and `uploads` stores PDFs. Back up both folders. For internet access, place it behind HTTPS and do not expose port 3000 directly to the public internet.

## AI library assistant

A floating **Ask AI** button opens a chat box that answers questions using only the PDFs in the `uploads` folder. Answers list the documents they came from, and each source links straight to the PDF. Within one chat session the assistant remembers the last few questions and answers, so short follow-ups like *"and its torque limit?"* are understood as continuations of the earlier question.

### Turn it on

The assistant uses Google Gemini, whose free tier covers both the chat model and the embeddings. No GPU or extra server memory is needed.

1. Create a free API key at <https://aistudio.google.com/apikey>.
2. Put it in a `.env` file next to `docker-compose.yml`:

   ```
   GEMINI_API_KEY=your-key-here
   ```

3. Restart: `docker compose up -d --build`.

Until a key is set the button still appears but reports that the assistant is switched off. Nothing else on the site changes.

### How it indexes

* Every PDF in `uploads` is read once, split into page aware passages and embedded. The first run happens a couple of seconds after start up and is paced to stay inside the free tier limits.
* Uploading a document through the Admin page indexes it immediately — the upload form shows live progress (including OCR page counts) until the document is searchable. Files copied straight into `uploads` are picked up by a background scan every 5 minutes.
* Editing a document's type re-checks it, so moving a document into a login-required type takes it out of the index again. The assistant only ever answers from **public** document types, never from `Drawings`.
* Scanned PDFs (no text layer) are **read with OCR automatically** — pages are rasterised and processed with Tesseract so the assistant can answer from scans too. Hybrid PDFs that mix text pages with scanned pages (common in large drawing sets) have only their scanned pages recovered. See the OCR settings below.

### How it searches

Retrieval is **hybrid**: every question runs through two searches and the results are fused.

* **Semantic search** (embeddings) finds passages that *mean* the same thing as the question, even in different words.
* **Keyword search** (SQLite FTS5, bm25) finds passages containing the *exact terms* asked for — document numbers, named limits, part names. Small embedding models often rank merely similar-looking text above the passage that actually holds the answer; the keyword half fixes exactly that.

A passage strong in both signals wins, and the citation (document + page) the model shows is taken from the passages it actually received. Follow-up questions are first rewritten into a self-contained form (using the previous turn) so pronouns like "it" still retrieve the right pages.

Useful overrides (set them alongside `GEMINI_API_KEY`):

| Setting | Default | Purpose |
| --- | --- | --- |
| `GEMINI_CHAT_MODEL` | `gemini-3.8-flash` | Chat model. Falls back automatically if the name is retired. |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | Embedding model. Changing it needs a re-index. |
| `RAG_TOP_K` | `6` | Passages given to the model per question. |
| `RAG_RESCAN_INTERVAL_MS` | `300000` | How often `uploads` is rescanned. |
| `RAG_MIN_SCORE` | `0.3` | Similarity needed before a passage counts as a match. |
| `RAG_KEYWORD_WEIGHT` | `0.5` | Weight of the exact-keyword (FTS5/bm25) signal in retrieval ranking. |
| `RAG_VECTOR_WEIGHT` | `0.5` | Weight of the semantic (embedding) signal in retrieval ranking. |
| `RAG_PER_DOC_LIMIT` | `3` | At most this many passages per document are retrieved, so one document cannot crowd out others that may hold the answer. |
| `RAG_RETRIEVE_POOL` | `4 × RAG_TOP_K` | How many candidate passages are scored before the best are picked. |
| `RAG_HISTORY_TURNS` | `2` local, `4` hosted | How many earlier questions and answers from the same chat are given to the model. |
| `RAG_HISTORY_CHARS` | `1200` local, `2400` hosted | Character budget for that conversation history. |

After changing the embedding model, clear the old vectors and rebuild:

```
docker compose exec rail-docs node -e "require('better-sqlite3')(process.env.DB_PATH).exec('DELETE FROM rag_chunks; DELETE FROM rag_documents;')"
docker compose restart rail-docs
```

### OCR settings (scanned PDFs)

Scanned PDFs carry no selectable text, so their pages are rasterised and read
with Tesseract (runs as WebAssembly inside Node — no system packages needed).
OCR only runs for files that have no text layer, so normal PDFs are unaffected.
Already-skipped scans are picked up automatically by the 5-minute rescan after
the first deploy with this change.

| Setting | Default | Purpose |
| --- | --- | --- |
| `RAG_OCR` | `1` | Set to `0` to turn OCR off. |
| `RAG_OCR_LANGS` | `eng` | Tesseract languages, e.g. `eng+hin`. Language data downloads once into the cache folder (a few MB per language). |
| `RAG_OCR_LANG_PATH` | *(download)* | Folder with `<lang>.traineddata.gz` files for fully offline servers. |
| `RAG_OCR_CACHE_DIR` | `./.ocr-cache` | Where downloaded language data is cached (mounted at `/app/ocr-cache` in Docker). |
| `RAG_OCR_MAX_PAGES` | `40` | OCR gives up after this many pages of one PDF to bound CPU time. |
| `RAG_OCR_DPI_SCALE` | `2` | Render scale; raise to 3 for small print, lower to 1.5 for large clear text. |
| `RAG_OCR_WORKERS` | *(one per core minus one)* | Parallel OCR workers. Indexing runs one worker per core by default so scans read as fast as the CPU allows; set a number (e.g. `4`) to keep cores free for other work. |

On the bundled Ollama setup, `OLLAMA_NUM_THREADS` (default 8) caps how many
threads a single AI question may use, so answering, indexing and the web app
can share the CPU instead of fighting over it.

## Local run

`npm install`

`$env:ADMIN_PASSWORD='your-password'; npm start` (PowerShell)

The app can be installed from the browser using its Add to Home Screen option. iPhone installation requires Safari; Android installation works in Chrome.
