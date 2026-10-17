"""RRF fusion and metric maths are pure functions (no DB). Hybrid retrieval is also checked against
a real Postgres when available, using a deterministic fake embedder."""
import hashlib
import uuid

import numpy as np
import psycopg
import pytest

from docqa.config import load_settings
from docqa.eval_retrieval import evaluate, rank_metrics, to_markdown
from docqa.ingest import ingest
from docqa.loader import load_corpus
from docqa.retrieval import Retriever, rrf_fuse
from docqa.store import Hit, Store

S = load_settings()


def H(cid, doc="d", score=0.5):
    return Hit(cid, doc, "T", "T", "text", "u", "current", "public", score)


def test_rrf_rewards_agreement_between_searches():
    fused = rrf_fuse([H("a"), H("b"), H("c")], [H("c"), H("d"), H("a")], rrf_k=60, top_k=4)
    ids = [r.chunk_id for r in fused]
    assert set(ids[:2]) == {"a", "c"}  # found by both searches
    assert ids[0] == "a"  # rank 1 + rank 3 beats rank 3 + rank 1 on the tie-break (best single rank)
    assert fused[0].vector_rank == 1 and fused[0].keyword_rank == 3


def test_rrf_respects_top_k_and_is_deterministic():
    v, k = [H(f"v{i}") for i in range(10)], [H(f"k{i}") for i in range(10)]
    a, b = rrf_fuse(v, k, top_k=5), rrf_fuse(v, k, top_k=5)
    assert len(a) == 5 and [r.chunk_id for r in a] == [r.chunk_id for r in b]


def test_rrf_with_one_empty_list_keeps_order():
    assert [r.chunk_id for r in rrf_fuse([H("x"), H("y")], [], top_k=5)] == ["x", "y"]


def test_rank_metrics():
    m = rank_metrics(["a", "b", "c", "d"], {"c", "z"}, (1, 3, 5))
    assert m["hit@1"] == 0 and m["hit@3"] == 1 and m["recall@3"] == 0.5 and abs(m["mrr"] - 1 / 3) < 1e-9
    assert rank_metrics(["a"], {"q"}, (1,))["mrr"] == 0.0


class StubRetriever:
    def __init__(self, ranked):
        self.ranked = ranked

    def retrieve(self, query, *, k, mode, query_vec=None, **_):
        from docqa.retrieval import RetrievalResult, Retrieved
        items = [Retrieved(H(f"{d}:0", d), 1.0, vector_score=0.8) for d in self.ranked[query][:k]]
        return RetrievalResult(items, 0.8 if items else None, None)


def test_evaluate_aggregates_and_flags_misses():
    qs = [
        {"id": "q1", "question": "a?", "type": "factual", "answerable": True, "sources": ["x"]},
        {"id": "q2", "question": "b?", "type": "factual", "answerable": True, "sources": ["y"]},
        {"id": "q3", "question": "c?", "type": "unanswerable", "answerable": False, "sources": []},
    ]
    r = StubRetriever({"a?": ["x", "z"], "b?": ["z", "w"], "c?": ["z"]})
    rep = evaluate(r, qs, modes=("hybrid",), ks=(1, 5))
    e = rep["modes"]["hybrid"]
    assert e["scored_questions"] == 2 and e["metrics"]["hit@5"] == 0.5 and e["metrics"]["mrr"] == 0.5
    assert [f["id"] for f in e["failures"]] == ["q2"] and e["policy_violations"] == 0
    assert "vector_top1_cosine" in e and "hit@5" in to_markdown(rep)


def _db_ok():
    try:
        psycopg.connect(S.database_url, connect_timeout=2).close()
        return True
    except Exception:  # noqa: BLE001
        return False


class FakeEmbedder:
    model = "fake-8"

    def embed(self, texts):
        out = []
        for t in texts:
            v = np.random.RandomState(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).randn(8)
            out.append((v / np.linalg.norm(v)).tolist())
        return out

    def embed_query(self, text):
        return self.embed([text])[0]


@pytest.mark.skipif(not _db_ok(), reason="Postgres not reachable (run: docker compose up -d)")
def test_hybrid_retrieval_on_real_database():
    st = Store(S.database_url, f"docqa_test_{uuid.uuid4().hex[:8]}")
    try:
        st.init_schema(8, "fake-8")
        emb = FakeEmbedder()
        ingest(load_corpus(S.corpus_dir), emb, st)
        rt = Retriever(st, emb, candidate_k=5)
        q = "How long does a refund take to reach my card?"
        res = rt.retrieve(q, k=5, mode="hybrid")
        assert all(r.hit.audience == "public" and r.hit.status == "current" for r in res.items)
        top_keyword = st.keyword_search(q, 1)[0].chunk_id
        assert top_keyword in [r.chunk_id for r in res.items]  # keyword evidence survives fusion
        assert res.top_vector_score is not None
        # forbidden documents are impossible to retrieve in any mode
        for mode in ("vector", "keyword", "hybrid"):
            got = rt.retrieve("goodwill refund exception $100 support agents", k=10, mode=mode).items
            assert not any(r.doc_id in {"internal-escalation-runbook", "refund-policy-v1"} for r in got)
    finally:
        st.drop_schema()
        st.close()


def test_keyword_weight_zero_lets_vector_order_win():
    fused = rrf_fuse([H("v1"), H("v2")], [H("k1"), H("k2")], top_k=4, keyword_weight=0.0)
    assert [r.chunk_id for r in fused[:2]] == ["v1", "v2"]
