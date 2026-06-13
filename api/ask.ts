// RAG endpoint — serverless replacement for server.py's /ask.
//
// Flow: rate-limit -> embed the question (OpenAI) -> cosine top-5 over the
// bundled memoir vectors -> synthesize with gpt-4o-mini. Response shape and
// prompt are identical to the original FastAPI version.

import type { VercelRequest, VercelResponse } from "@vercel/node";
import OpenAI from "openai";
import { retrieve, EMBEDDING_MODEL, type ScoredChunk } from "../lib/retrieve.js";
import { checkRateLimit } from "../lib/ratelimit.js";

const MIN_CHUNK_CHARS = 50;
const CHAT_MODEL = "gpt-4o-mini";

const SYSTEM_INSTRUCTION = `You are an expert World War II historian. You are receiving a question about Captain James V. Morgia. Use the provided memoir excerpts to answer.
Style:
- Tone: Authoritative, warm, narrative.
- First mention: 'Captain James V. Morgia'. Subsequent: 'Captain Jim'.
- Perspective: Third person (He/Him).
- Content: Synthesize the excerpts into a cohesive story.`;

// Trim an excerpt back to its last complete sentence (matches server.py).
function cleanExcerptText(text: string): string {
  const cutoff = Math.max(text.lastIndexOf("."), text.lastIndexOf("!"), text.lastIndexOf("?"));
  return cutoff !== -1 ? text.slice(0, cutoff + 1) : text;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    return res.status(405).json({ detail: "Method not allowed" });
  }

  if (!(await checkRateLimit(req))) {
    return res.status(429).json({ detail: "Too many requests. Please wait a minute." });
  }

  const question = (req.body?.question ?? "").toString().trim();
  if (!question) {
    return res.status(400).json({ detail: "Missing 'question'." });
  }

  if (!process.env.OPENAI_API_KEY) {
    return res.status(503).json({ detail: "AI System is not ready yet." });
  }

  try {
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

    const embedRes = await openai.embeddings.create({ model: EMBEDDING_MODEL, input: question });
    const queryEmbedding = embedRes.data[0].embedding;

    const nodes = retrieve(queryEmbedding, 5);
    const validNodes: ScoredChunk[] = nodes.filter((n) => n.text.trim().length >= MIN_CHUNK_CHARS);

    if (validNodes.length === 0) {
      return res.status(200).json({
        summary:
          "I searched the archives but couldn't find specific details on that topic in Captain Jim's memoir.",
        excerpts: [],
      });
    }

    const contextText = validNodes.map((n) => `Excerpt: ${n.text}`).join("\n\n");

    const completion = await openai.chat.completions.create({
      model: CHAT_MODEL,
      temperature: 0.3,
      messages: [
        { role: "system", content: SYSTEM_INSTRUCTION },
        { role: "user", content: `Context:\n${contextText}\n\nQuestion: ${question}` },
      ],
    });

    const summary = completion.choices[0].message.content;

    const excerpts = validNodes.slice(0, 3).map((n) => ({
      text: cleanExcerptText(n.text),
      chapter: n.source,
    }));

    return res.status(200).json({ summary, excerpts });
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    console.error("ask error:", message);
    return res.status(500).json({ detail: message });
  }
}
