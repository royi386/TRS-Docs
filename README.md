# Rail Docs

Searchable Indian Railways RDSO SMI, MS and TC PDF library. Engineers can search by document number or caption and download matching PDFs. The `Admin` page requires the server-side `ADMIN_PASSWORD` and is the only way to upload documents.

## Run with Docker on OMV

1. Copy this folder to the OMV server at `10.189.34.56`.
2. Edit `docker-compose.yml` and replace `change-this-to-a-long-password`.
3. Start it with `docker compose up -d --build`.
4. Open `http://10.189.34.56:3000` on the railway network.

The `data` folder stores SQLite metadata and `uploads` stores PDFs. Back up both folders. For internet access, place it behind HTTPS and do not expose port 3000 directly to the public internet.

## Local run

`npm install`

`$env:ADMIN_PASSWORD='your-password'; npm start` (PowerShell)

The app can be installed from the browser using its Add to Home Screen option. iPhone installation requires Safari; Android installation works in Chrome.
