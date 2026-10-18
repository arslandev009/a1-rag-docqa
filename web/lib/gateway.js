// Edge gateway for the A1 demo (Cloudflare Pages Functions). No dependencies.
//
// Order of defence: (1) Cloudflare Turnstile proves a human opened the page (bots stop here, the Render backend
// is never woken); (2) a short-lived signed cookie, so the visitor is not challenged on every message;
// (3) only three allow-listed routes are forwarded, with the gateway secret and the real client IP added;
// (4) the backend applies its own per-IP limit, daily budget and input caps.
//
// Environment (set in the Pages dashboard, never in the repo):
//   TURNSTILE_SECRET  Turnstile secret key            SESSION_SECRET  long random string (signs the cookie)
//   API_ORIGIN        https://<service>.onrender.com   API_PREFIX      "/a1" in production, "" for the local API
//   GATEWAY_SECRET    must match the backend's GATEWAY_SECRET (empty = header not sent, local development only)

const SESSION_TTL_S = 30 * 60;
const MAX_BODY_CHARS = 2000;
const COOKIE = "demo_session";
const TURNSTILE_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify";

const ROUTES = {
  sources: { method: "GET", path: "/sources", timeoutMs: 60_000 },
  chat: { method: "POST", path: "/chat", timeoutMs: 60_000 },
  warmup: { method: "GET", path: "/health", timeoutMs: 100_000 }, // a cold Render instance needs about a minute
};

const enc = new TextEncoder();

function json(status, body, headers = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store", ...headers },
  });
}

async function hmacHex(secret, message) {
  const key = await crypto.subtle.importKey("raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(message));
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function safeEqual(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

function readCookie(request, name) {
  for (const part of (request.headers.get("cookie") || "").split(";")) {
    const [k, ...v] = part.trim().split("=");
    if (k === name) return v.join("=");
  }
  return "";
}

export async function makeSessionCookie(env, nowMs, secure) {
  const exp = Math.floor(nowMs / 1000) + SESSION_TTL_S;
  const sig = await hmacHex(env.SESSION_SECRET, String(exp));
  return `${COOKIE}=${exp}.${sig}; HttpOnly; SameSite=Lax; Path=/api; Max-Age=${SESSION_TTL_S}${secure ? "; Secure" : ""}`;
}

export async function validSession(request, env, nowMs) {
  const [exp, sig] = readCookie(request, COOKIE).split(".");
  if (!exp || !sig || !/^\d+$/.test(exp) || Number(exp) * 1000 < nowMs) return false;
  return safeEqual(sig, await hmacHex(env.SESSION_SECRET, exp));
}

async function createSession(request, env, url, nowMs, doFetch) {
  if (request.method !== "POST") return json(405, { detail: "Method not allowed." }, { allow: "POST" });
  let token = "";
  try {
    token = (await request.json()).token;
  } catch {
    /* handled below */
  }
  if (typeof token !== "string" || !token || token.length > 2048) return json(400, { detail: "Missing verification token." });

  const form = new URLSearchParams({ secret: env.TURNSTILE_SECRET, response: token });
  const ip = request.headers.get("cf-connecting-ip");
  if (ip) form.set("remoteip", ip);
  let success = false;
  try {
    const r = await doFetch(TURNSTILE_URL, { method: "POST", body: form });
    success = (await r.json()).success === true;
  } catch {
    return json(502, { detail: "Could not reach the verification service. Please try again." });
  }
  if (!success) return json(403, { detail: "Verification failed. Please reload the page." });
  return json(200, { ok: true, ttl: SESSION_TTL_S }, { "set-cookie": await makeSessionCookie(env, nowMs, url.protocol === "https:") });
}

export async function handle(route, request, env, deps = {}) {
  const nowMs = deps.now ? deps.now() : Date.now();
  const doFetch = deps.fetch || fetch;
  if (!env.SESSION_SECRET || !env.TURNSTILE_SECRET || !env.API_ORIGIN) return json(500, { detail: "The gateway is not configured." });

  const url = new URL(request.url);
  const origin = request.headers.get("origin");
  if (origin) {
    let sameOrigin = false;
    try {
      sameOrigin = new URL(origin).host === url.host;
    } catch {
      /* "null" and malformed origins are rejected */
    }
    if (!sameOrigin) return json(403, { detail: "Cross-origin requests are not allowed." });
  }

  if (route === "session") return createSession(request, env, url, nowMs, doFetch);

  const spec = ROUTES[route];
  if (!spec) return json(404, { detail: "Not found." });
  if (request.method !== spec.method) return json(405, { detail: "Method not allowed." }, { allow: spec.method });
  if (!(await validSession(request, env, nowMs))) return json(401, { detail: "Session expired. Please verify again." });

  let body;
  if (spec.method === "POST") {
    const raw = await request.text();
    if (raw.length > MAX_BODY_CHARS) return json(413, { detail: "Request too large." });
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch {
      return json(400, { detail: "Body must be JSON." });
    }
    if (!parsed || typeof parsed.question !== "string") return json(400, { detail: "Missing question." });
    const clean = { question: parsed.question };
    if (typeof parsed.provider === "string") clean.provider = parsed.provider; // nothing else is forwarded
    body = JSON.stringify(clean);
  }

  const headers = { "content-type": "application/json" };
  if (env.GATEWAY_SECRET) headers["x-gateway-secret"] = env.GATEWAY_SECRET;
  const ip = request.headers.get("cf-connecting-ip");
  if (ip) headers["x-real-client-ip"] = ip;

  const target = `${env.API_ORIGIN.replace(/\/+$/, "")}${env.API_PREFIX || ""}${spec.path}`;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), spec.timeoutMs);
  try {
    const upstream = await doFetch(target, { method: spec.method, headers, body, signal: ctrl.signal });
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "content-type": upstream.headers.get("content-type") || "application/json", "cache-control": "no-store" },
    });
  } catch {
    return json(504, { detail: "The demo backend is waking up (about 1 minute on free hosting) or unavailable. Please try again shortly." });
  } finally {
    clearTimeout(timer);
  }
}
