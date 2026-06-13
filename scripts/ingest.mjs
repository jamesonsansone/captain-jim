// Re-embeds Captain Jim's memoir chunks using the OpenAI Embeddings API.
//
// Reads the 120 chunks already tuned in storage/docstore.json (so chunking
// stays identical to the old LlamaIndex/FastEmbed pipeline), embeds each one
// with text-embedding-3-small, and writes lib/embeddings.json for the
// serverless /api/ask function to load in memory.
//
// Run: node --env-file=.env scripts/ingest.mjs

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import OpenAI from "openai";

const EMBEDDING_MODEL = "text-embedding-3-small";
const BATCH_SIZE = 100; // OpenAI accepts arrays; keep batches modest.

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, "..");
const DOCSTORE = path.join(ROOT, "storage", "docstore.json");
const OUT_DIR = path.join(ROOT, "lib");
const OUT_FILE = path.join(OUT_DIR, "embeddings.json");

function loadChunks() {
  const raw = JSON.parse(fs.readFileSync(DOCSTORE, "utf8"));
  const data = raw["docstore/data"] || {};
  const chunks = [];
  for (const [id, entry] of Object.entries(data)) {
    const node = entry.__data__;
    if (!node || typeof node.text !== "string") continue;
    const text = node.text.trim();
    if (!text) continue;
    chunks.push({
      id,
      text,
      source: node.metadata?.file_name || "Memoir Archive",
    });
  }
  return chunks;
}

async function embedBatch(client, inputs) {
  const res = await client.embeddings.create({
    model: EMBEDDING_MODEL,
    input: inputs,
  });
  // Responses come back in the same order as inputs.
  return res.data.map((d) => d.embedding);
}

async function main() {
  if (!process.env.OPENAI_API_KEY) {
    console.error("❌ OPENAI_API_KEY missing. Run with: node --env-file=.env scripts/ingest.mjs");
    process.exit(1);
  }

  const client = new OpenAI();
  const chunks = loadChunks();
  console.log(`🔹 Loaded ${chunks.length} chunks from docstore.json`);

  const records = [];
  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE);
    const embeddings = await embedBatch(client, batch.map((c) => c.text));
    batch.forEach((c, j) => {
      records.push({ id: c.id, text: c.text, source: c.source, embedding: embeddings[j] });
    });
    console.log(`   Embedded ${Math.min(i + BATCH_SIZE, chunks.length)}/${chunks.length}`);
  }

  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(OUT_FILE, JSON.stringify({ model: EMBEDDING_MODEL, records }));

  const dims = records[0]?.embedding.length ?? 0;
  const sizeMb = (fs.statSync(OUT_FILE).size / 1e6).toFixed(2);
  console.log(`✅ Wrote ${records.length} vectors (${dims} dims) -> lib/embeddings.json (${sizeMb} MB)`);
}

main().catch((e) => {
  console.error("❌ Ingest failed:", e);
  process.exit(1);
});
