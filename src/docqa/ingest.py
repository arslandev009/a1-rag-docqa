"""Idempotent ingestion: only changed chunks are re-embedded."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

from .chunker import chunk_document
from .loader import Document
from .store import Store


class _Embedder(Protocol):
    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass
class IngestReport:
    docs: int = 0
    chunks: int = 0
    embedded: int = 0
    unchanged: int = 0
    deleted_chunks: int = 0
    deleted_docs: int = 0
    seconds: float = 0.0


def content_hash(text: str, model: str) -> str:
    """Hash includes the model so switching embedding models re-embeds everything."""
    return hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()


def ingest(docs: Sequence[Document], embedder: _Embedder, store: Store, *,
           max_chars: int = 1200, overlap_sentences: int = 1, prune: bool = False) -> IngestReport:
    t0 = time.perf_counter()
    rep = IngestReport(docs=len(docs))
    for doc in docs:
        chunks = chunk_document(doc, max_chars, overlap_sentences)
        existing = store.chunk_hashes(doc.doc_id)
        todo, same = [], []
        for c in chunks:
            h = content_hash(c.text, embedder.model)
            (same if existing.get(c.chunk_id) == h else todo).append((c, h))
        vectors = embedder.embed([c.text for c, _ in todo]) if todo else []  # network call outside the DB transaction
        stale = set(existing) - {c.chunk_id for c in chunks}
        with store.transaction():
            store.upsert_document(doc)
            for (c, h), v in zip(todo, vectors):
                store.upsert_chunk(c, v, h)
            for c, _ in same:
                store.update_chunk_meta(c)
            rep.deleted_chunks += store.delete_chunks(sorted(stale))
        rep.chunks += len(chunks)
        rep.embedded += len(todo)
        rep.unchanged += len(same)
    if prune:
        rep.deleted_docs = store.delete_docs_not_in([d.doc_id for d in docs])
    rep.seconds = time.perf_counter() - t0
    return rep
