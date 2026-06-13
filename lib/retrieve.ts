// In-memory vector search over Captain Jim's memoir.
//
// The corpus is tiny (120 chunks), so we bundle the precomputed OpenAI
// embeddings and do a plain cosine scan per query — microseconds, no DB.

import embeddingsData from "./embeddings.json";

export interface MemoirChunk {
  id: string;
  text: string;
  source: string;
  embedding: number[];
}

interface EmbeddingsFile {
  model: string;
  records: MemoirChunk[];
}

const { model: EMBEDDING_MODEL, records: CHUNKS } = embeddingsData as EmbeddingsFile;

export { EMBEDDING_MODEL };

export interface ScoredChunk extends MemoirChunk {
  score: number;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/** Returns the top-k chunks by cosine similarity to the query embedding. */
export function retrieve(queryEmbedding: number[], topK = 5): ScoredChunk[] {
  return CHUNKS.map((chunk) => ({ ...chunk, score: cosineSimilarity(queryEmbedding, chunk.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}
