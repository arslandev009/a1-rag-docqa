"""Hybrid retrieval: vector search + keyword search fused with Reciprocal Rank Fusion (RRF).

RRF scores each chunk as sum(1 / (rrf_k + rank)) over the lists it appears in. It needs no score
calibration between the two searches (cosine similarity and ts_rank live on different scales), and
a chunk that both searches like rises to the top. Access rules (audience/status) are applied inside
the SQL of both searches, so nothing forbidden is ever a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .store import Hit, Store

MODES = ("vector", "keyword", "hybrid")


@dataclass(frozen=True)
class Retrieved:
    hit: Hit
    score: float  # fused RRF score (hybrid) or the raw score of the single search used
    vector_rank: int | None = None
    keyword_rank: int | None = None
    vector_score: float | None = None  # cosine similarity: the calibrated signal used later for abstention

    @property
    def chunk_id(self) -> str:
        return self.hit.chunk_id

    @property
    def doc_id(self) -> str:
        return self.hit.doc_id


@dataclass(frozen=True)
class RetrievalResult:
    items: list[Retrieved]
    top_vector_score: float | None  # best cosine among vector candidates (even if not selected)
    top_keyword_score: float | None


def rrf_fuse(vector_hits: Sequence[Hit], keyword_hits: Sequence[Hit], *, rrf_k: int = 60, top_k: int = 5,
             keyword_weight: float = 1.0) -> list[Retrieved]:
    scores: dict[str, float] = {}
    info: dict[str, dict] = {}
    for source, hits in (("vector", vector_hits), ("keyword", keyword_hits)):
        for rank, h in enumerate(hits, start=1):
            e = info.setdefault(h.chunk_id, {"hit": h, "vector_rank": None, "keyword_rank": None, "vector_score": None})
            e[f"{source}_rank"] = rank
            if source == "vector":
                e["vector_score"] = h.score
            weight = keyword_weight if source == "keyword" else 1.0
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + weight / (rrf_k + rank)

    def best_rank(cid: str) -> int:
        return min(r for r in (info[cid]["vector_rank"], info[cid]["keyword_rank"]) if r is not None)

    order = sorted(scores, key=lambda cid: (-scores[cid], best_rank(cid), cid))[:top_k]
    return [Retrieved(info[c]["hit"], scores[c], info[c]["vector_rank"], info[c]["keyword_rank"], info[c]["vector_score"])
            for c in order]


class Retriever:
    """candidate_k = how many results each search contributes before fusion. Keep it small relative to
    the corpus (this demo has only 35 public chunks); use 30-50 for corpora of thousands of chunks."""

    def __init__(self, store: Store, embedder=None, *, candidate_k: int = 10, rrf_k: int = 60,
                 keyword_weight: float = 1.0) -> None:
        self.store, self.embedder = store, embedder
        self.candidate_k, self.rrf_k, self.keyword_weight = candidate_k, rrf_k, keyword_weight

    def retrieve(self, query: str, *, k: int = 5, mode: str = "hybrid",
                 audiences: Sequence[str] = ("public",), statuses: Sequence[str] = ("current",),
                 query_vec: Sequence[float] | None = None) -> RetrievalResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        v_hits: list[Hit] = []
        k_hits: list[Hit] = []
        if mode in ("vector", "hybrid"):
            vec = query_vec if query_vec is not None else self.embedder.embed_query(query)
            v_hits = self.store.vector_search(vec, self.candidate_k, audiences, statuses)
        if mode in ("keyword", "hybrid"):
            k_hits = self.store.keyword_search(query, self.candidate_k, audiences, statuses)

        if mode == "hybrid":
            items = rrf_fuse(v_hits, k_hits, rrf_k=self.rrf_k, top_k=k, keyword_weight=self.keyword_weight)
        elif mode == "vector":
            items = [Retrieved(h, h.score, vector_rank=i, vector_score=h.score) for i, h in enumerate(v_hits[:k], 1)]
        else:
            items = [Retrieved(h, h.score, keyword_rank=i) for i, h in enumerate(k_hits[:k], 1)]
        return RetrievalResult(items, v_hits[0].score if v_hits else None, k_hits[0].score if k_hits else None)
