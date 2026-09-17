FROM node:24-bookworm-slim
WORKDIR /app
# fontconfig + at least one font family are required: without them the Linux
# build of @napi-rs/canvas silently draws nothing, which breaks PDF rasterising
# for OCR (pages come out blank).
RUN apt-get update \
	&& apt-get install -y --no-install-recommends python3 make g++ fontconfig fonts-dejavu-core \
	&& rm -rf /var/lib/apt/lists/*
COPY package*.json ./
RUN apt-get update \
	&& apt-get install -y --no-install-recommends python3 make g++ \
	&& npm ci --omit=dev \
	&& apt-get purge -y --auto-remove python3 make g++ \
	&& rm -rf /var/lib/apt/lists/*
COPY . .
RUN mkdir -p /app/data /app/uploads
ENV NODE_ENV=production
ENV PORT=3000
ENV DB_PATH=/app/data/smi_tc.db
ENV UPLOAD_DIR=/app/uploads
EXPOSE 3000
CMD ["node", "server.js"]
