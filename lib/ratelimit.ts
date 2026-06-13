// Per-IP rate limiting via Upstash Redis — the serverless-native replacement
// for the old slowapi limiter (5 requests / minute), shared by both routes.

import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";
import type { VercelRequest } from "@vercel/node";

// Disabled gracefully if the Upstash env vars aren't set (e.g. local dev),
// so the app still runs — it just won't throttle.
const enabled = Boolean(
  process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN,
);

const limiter = enabled
  ? new Ratelimit({
      redis: Redis.fromEnv(),
      limiter: Ratelimit.slidingWindow(5, "1 m"),
      prefix: "captain-jim",
    })
  : null;

/** Mirrors server.py: trust the first IP in x-forwarded-for. */
export function getClientIp(req: VercelRequest): string {
  const forwarded = req.headers["x-forwarded-for"];
  const value = Array.isArray(forwarded) ? forwarded[0] : forwarded;
  if (value) return value.split(",")[0].trim();
  return "127.0.0.1";
}

/** Returns true if the request is allowed, false if it should be rejected (429). */
export async function checkRateLimit(req: VercelRequest): Promise<boolean> {
  if (!limiter) return true;
  const { success } = await limiter.limit(getClientIp(req));
  return success;
}
