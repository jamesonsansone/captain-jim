// ElevenLabs TTS proxy — serverless replacement for server.py's /speak.
// Same voice settings, model, and Beho -> Bay-ho phonetic fix as the original.

import type { VercelRequest, VercelResponse } from "@vercel/node";
import { checkRateLimit } from "../lib/ratelimit";

const PHONETIC_CORRECTIONS: Record<string, string> = {
  Beho: "Bay-ho",
  beho: "bay-ho",
};

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    return res.status(405).json({ detail: "Method not allowed" });
  }

  if (!(await checkRateLimit(req))) {
    return res.status(429).json({ detail: "Too many requests. Please wait a minute." });
  }

  const voiceId = process.env.ELEVENLABS_VOICE_ID;
  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!voiceId || !apiKey) {
    return res.status(500).json({ detail: "Audio configuration missing." });
  }

  let textToSpeak = (req.body?.text ?? "").toString();
  if (!textToSpeak.trim()) {
    return res.status(400).json({ detail: "Missing 'text'." });
  }

  for (const [word, phonetic] of Object.entries(PHONETIC_CORRECTIONS)) {
    textToSpeak = textToSpeak.split(word).join(phonetic);
  }

  try {
    const response = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
      {
        method: "POST",
        headers: { "xi-api-key": apiKey, "Content-Type": "application/json" },
        body: JSON.stringify({
          text: textToSpeak,
          model_id: "eleven_multilingual_v2",
          voice_settings: {
            stability: 0.35,
            similarity_boost: 0.95,
            style: 0.2,
            speed: 0.77,
            use_speaker_boost: true,
          },
        }),
      },
    );

    // Pass ElevenLabs' bytes straight through, as the original did. The frontend
    // already detects a tiny (<1KB) body as a quota/error response.
    const audio = Buffer.from(await response.arrayBuffer());
    res.setHeader("Content-Type", "audio/mpeg");
    return res.status(200).send(audio);
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    console.error("speak error:", message);
    return res.status(500).json({ detail: message });
  }
}
