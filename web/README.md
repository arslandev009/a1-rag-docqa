# A1 demo UI (Cloudflare Pages)

Static chat page for the Ledgerly Support Assistant plus a small edge gateway (Pages Functions). No build step and no
runtime dependencies: `public/` is the site, `functions/` and `lib/` are the gateway.

```
browser -> /api/session  Turnstile check, sets a signed 30-minute cookie
        -> /api/warmup   wakes the backend (cold start about 1 minute on Render free)
        -> /api/sources  public documents the assistant can cite
        -> /api/chat     question -> answer with citations
              |  gateway adds X-Gateway-Secret + the real client IP, forwards only {question, provider}
              v
           demo-api (FastAPI, /a1 router) -> Neon pgvector
```
The backend applies its own per-IP limit, daily budget and input caps; the gateway only lets three routes through.

## Local development
1. Start the API (from `projects/a1-rag-docqa`): `docker compose --profile app up -d` (listens on 127.0.0.1:8000, no gateway secret).
2. In this folder: `copy .dev.vars.example .dev.vars`, `npm install`, `npm run dev`, open http://localhost:8788.
   The test Turnstile keys in `config.js` and `.dev.vars.example` always pass.
3. Tests: `npm test` (gateway logic, no network).

## Deploy (Cloudflare Pages, free)
1. Create a Turnstile widget (Dashboard -> Turnstile) for the `*.pages.dev` hostname; put the **site key** in `public/config.js`.
2. Pages -> Create project -> connect the repo. Root directory: `web`. Build command: none. Build output directory: `public`.
3. Environment variables / secrets (Production), never in the repo:
   `TURNSTILE_SECRET`, `SESSION_SECRET` (long random string), `GATEWAY_SECRET` (same value as the backend),
   `API_ORIGIN` (`https://<service>.onrender.com`), `API_PREFIX` (`/a1`).
4. Fill in `videoUrl` in `config.js` once the walkthrough video exists (the link stays hidden while it is empty).

Security notes: the CSP allows only this origin and Cloudflare's Turnstile; model output is rendered with `textContent`
(never as HTML); `style-src` keeps `'unsafe-inline'` because the Turnstile widget needs it.
