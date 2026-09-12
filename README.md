# Rail Docs

Searchable Indian Railways RDSO SMI, MS and TC PDF library. Engineers can search by document number or caption and download matching PDFs. The `Admin` page requires the server-side `ADMIN_PASSWORD` and is the only way to upload documents.

## Run with Docker on OMV

1. Copy this folder to the OMV server at `10.189.34.56`.
2. Edit `docker-compose.yml` and replace `change-this-to-a-long-password`.
3. Start it with `docker compose up -d --build`.
4. Open `http://10.189.34.56:3000` on the railway network.

The `data` folder stores SQLite metadata and `uploads` stores PDFs. Back up both folders. For internet access, place it behind HTTPS and do not expose port 3000 directly to the public internet.

## AI library assistant

A floating **Ask AI** button opens a chat box that answers questions using only the PDFs in the `uploads` folder. Answers list the documents they came from, and each source links straight to the PDF.

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
* Uploading a document through the Admin page indexes it immediately.
* A background scan every 5 minutes picks up PDFs copied straight into `uploads`, and removes ones that were deleted.
* Editing a document's type re-checks it, so moving a document into a login-required type takes it out of the index again. The assistant only ever answers from **public** document types, never from `Drawings`.
* Scanned PDFs with no text layer are skipped and reported; run them through OCR first if you want them searchable.

Useful overrides (set them alongside `GEMINI_API_KEY`):

| Setting | Default | Purpose |
| --- | --- | --- |
| `GEMINI_CHAT_MODEL` | `gemini-3.8-flash` | Chat model. Falls back automatically if the name is retired. |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | Embedding model. Changing it needs a re-index. |
| `RAG_TOP_K` | `6` | Passages given to the model per question. |
| `RAG_RESCAN_INTERVAL_MS` | `300000` | How often `uploads` is rescanned. |
| `RAG_MIN_SCORE` | `0.3` | Similarity needed before a passage counts as a match. |

After changing the embedding model, clear the old vectors and rebuild:

```
docker compose exec rail-docs node -e "require('better-sqlite3')(process.env.DB_PATH).exec('DELETE FROM rag_chunks; DELETE FROM rag_documents;')"
docker compose restart rail-docs
```

## Local run

`npm install`

`$env:ADMIN_PASSWORD='your-password'; npm start` (PowerShell)

The app can be installed from the browser using its Add to Home Screen option. iPhone installation requires Safari; Android installation works in Chrome.
