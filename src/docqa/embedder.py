"""Embeddings through any OpenAI-compatible /embeddings endpoint (Gemini, OpenAI, local llama-server).

Provider keys and base URLs come from the shared `llm-client` provider registry; the embedding
model id comes from EMBED_MODEL. Vectors are L2-normalised so cosine distance is well behaved
regardless of provider.
"""
from __future__ import annotations

import os
import random
import time
from typing import Any, Sequence

import numpy as np
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


class Embedder:
    def __init__(
        self,
        provider: str,
        model: str,
        *,
        batch_size: int = 32,
        max_retries: int = 4,
        _client: Any = None,  # injected in tests
    ) -> None:
        if not model:
            raise RuntimeError("EMBED_MODEL is empty. Set the exact embedding model id in .env (e.g. gemini-embedding-001).")
        self.provider, self.model = provider, model
        self.batch_size, self.max_retries = max(1, batch_size), max_retries
        if _client is None:
            from llm_client.providers import resolve  # shared key/base-url registry

            base_url, api_key, _ = resolve(provider, model)
            base_url = os.getenv("EMBED_BASE_URL") or base_url
            _client = OpenAI(base_url=base_url, api_key=api_key, timeout=60)
        self._client = _client

    def _call(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.embeddings.create(model=self.model, input=batch)
                # Some providers (Gemini's OpenAI-compatible endpoint) return index=None: keep API order then.
                ordered = sorted(
                    enumerate(resp.data),
                    key=lambda p: p[0] if getattr(p[1], "index", None) is None else p[1].index,
                )
                return [d.embedding for _, d in ordered]
            except Exception as exc:  # noqa: BLE001
                if not _retryable(exc) or attempt == self.max_retries:
                    raise
                time.sleep(min(2**attempt, 20) + random.random())
        raise RuntimeError("unreachable")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vecs = np.asarray(self._call(list(texts[i : i + self.batch_size])), dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out.extend((vecs / norms).tolist())
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def probe_dim(self) -> int:
        return len(self.embed_query("dimension probe"))
