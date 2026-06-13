// Local smoke test of the migrated RAG pipeline (no Vercel needed).
// Mirrors api/ask.ts: embed question -> cosine top-5 over lib/embeddings.json
// -> gpt-4o-mini synthesis. Run: node --env-file=.env scripts/test_rag.mjs "question"

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import OpenAI from "openai";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const { model: EMBEDDING_MODEL, records } = JSON.parse(
  fs.readFileSync(path.join(__dirname, "..", "lib", "embeddings.json"), "utf8"),
);

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

const SYSTEM = `You are an expert World War II historian. You are receiving a question about Captain James V. Morgia. Use the provided memoir excerpts to answer.
Style:
- Tone: Authoritative, warm, narrative.
- First mention: 'Captain James V. Morgia'. Subsequent: 'Captain Jim'.
- Perspective: Third person (He/Him).
- Content: Synthesize the excerpts into a cohesive story.`;

const question = process.argv[2] || "What happened at Beho?";
const openai = new OpenAI();

const emb = (await openai.embeddings.create({ model: EMBEDDING_MODEL, input: question })).data[0].embedding;
const top = records
  .map((r) => ({ ...r, score: cosine(emb, r.embedding) }))
  .sort((a, b) => b.score - a.score)
  .slice(0, 5);

console.log(`\n❓ ${question}\n`);
console.log("🔹 Top retrieved chunks (score / source / preview):");
top.forEach((t, i) => console.log(`  [${i + 1}] ${t.score.toFixed(3)}  ${t.source}  "${t.text.slice(0, 80).replace(/\n/g, " ")}..."`));

const ctx = top.map((t) => `Excerpt: ${t.text}`).join("\n\n");
const completion = await openai.chat.completions.create({
  model: "gpt-4o-mini",
  temperature: 0.3,
  messages: [
    { role: "system", content: SYSTEM },
    { role: "user", content: `Context:\n${ctx}\n\nQuestion: ${question}` },
  ],
});

console.log("\n📜 Summary:\n" + completion.choices[0].message.content + "\n");
