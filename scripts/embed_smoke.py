"""Embedding smoke test. Run from projects/a1-rag-docqa with the venv active:
    python scripts/embed_smoke.py
Needs in .env:  EMBED_PROVIDER (gemini|openai|local)  and  EMBED_MODEL (exact embedding model id).
Optional:       EMBED_BASE_URL (e.g. a separate llama-server started with --embedding)."""
import math
import os
import time

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from llm_client.providers import PROVIDERS, resolve

load_dotenv(find_dotenv(usecwd=True))

provider = os.getenv("EMBED_PROVIDER", "gemini")
model = os.getenv("EMBED_MODEL", "").strip()
if not model:
    raise SystemExit("EMBED_MODEL is empty. Put the exact embedding model id in .env (see README).")

base_url, api_key, _ = resolve(provider, model)  # reuses the provider's key + base URL
base_url = os.getenv("EMBED_BASE_URL") or base_url
client = OpenAI(base_url=base_url, api_key=api_key)

query = "How long does a refund take?"
docs = [
    "Approved refunds go back to the original payment method within 5 to 10 business days.",
    "Compute an HMAC-SHA256 of the raw request body with your endpoint secret.",
    "Growth includes 5 users and up to 500 invoices per month.",
]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


t0 = time.perf_counter()
try:
    resp = client.embeddings.create(model=model, input=[query, *docs])
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"Embedding call failed ({provider}/{model}): {exc!r}\n"
                     "404 = wrong model id, 401/403 = key, 429 = quota.")
elapsed = time.perf_counter() - t0

vecs = [d.embedding for d in resp.data]
print(f"provider={provider} model={model}")
print(f"dimension={len(vecs[0])}  latency={elapsed:.2f}s  vectors={len(vecs)}")
scores = [cosine(vecs[0], v) for v in vecs[1:]]
for s, d in sorted(zip(scores, docs), reverse=True):
    print(f"  {s:.3f}  {d[:70]}")
best = max(range(len(scores)), key=scores.__getitem__)
print("OK: refund sentence ranked first" if best == 0 else "WARNING: ranking looks wrong, check the model id")
