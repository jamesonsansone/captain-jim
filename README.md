# Captain Jim — WWII Memoir RAG

A RAG app trained on Captain James V. Morgia's WWII memoir, with ElevenLabs voice
narration. Runs entirely on Vercel (static frontend + serverless functions), $0/mo
hosting.

## Architecture

```
public/            Static UI (hand-built HTML/CSS/JS), served at /
  index.html, script.js, style.css, assets/
api/
  ask.ts           RAG: rate-limit -> embed question -> cosine top-5 -> gpt-4o-mini
  speak.ts         ElevenLabs text-to-speech proxy
lib/
  embeddings.json  120 memoir chunks pre-embedded with OpenAI (bundled, in-memory)
  retrieve.ts      Cosine top-k search
  ratelimit.ts     Per-IP limiter (Upstash Redis), 5 req/min
scripts/
  ingest.mjs       Re-embeds storage/docstore.json -> lib/embeddings.json
  test_rag.mjs     Local end-to-end RAG smoke test
```

There is no separate backend and no external vector database — the corpus is tiny
(120 chunks), so the embeddings are bundled and searched in memory.

## Environment variables (set in Vercel Project Settings)

- `OPENAI_API_KEY` — embeddings + chat completion
- `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` — voice narration
- `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` — rate limiting

## Re-ingesting the memoir

If the source content changes, regenerate the embeddings (requires `OPENAI_API_KEY`
in `.env`):

```bash
npm run ingest        # reads storage/docstore.json, writes lib/embeddings.json
npm run test_rag      # optional: sanity-check retrieval + synthesis locally
```

## Legacy

`scripts/server.py`, `query.py`, `ingest.py` and `chroma_db/` are the previous
Python/Render backend, kept for reference. They are no longer used or deployed.
