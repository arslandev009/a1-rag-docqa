"""Integration tests against a real Postgres + pgvector (docker compose up -d). Skipped if unreachable.
They use a throw-away schema and a deterministic fake embedder, so no API calls are made."""
import hashlib
import uuid
from dataclasses import replace

import numpy as np
import psycopg
import pytest

from docqa.chunker import chunk_document
from docqa.config import load_settings
from docqa.ingest import ingest
from docqa.loader import load_corpus
from docqa.store import SchemaMismatch, Store

S = load_settings()
DOCS = load_corpus(S.corpus_dir)
DIM = 8


def _db_ok() -> bool:
    try:
        psycopg.connect(S.database_url, connect_timeout=2).close()
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _db_ok(), reason="Postgres not reachable (run: docker compose up -d)")


class FakeEmbedder:
    model = "fake-8"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        out = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            v = np.random.RandomState(seed).randn(DIM)
            out.append((v / np.linalg.norm(v)).tolist())
        return out

    def embed_query(self, text):
        return self.embed([text])[0]


@pytest.fixture()
def store():
    st = Store(S.database_url, f"docqa_test_{uuid.uuid4().hex[:8]}")
    st.init_schema(DIM, "fake-8")
    yield st
    st.drop_schema()
    st.close()


def run(store, docs=DOCS, emb=None, **kw):
    return ingest(docs, emb or FakeEmbedder(), store, **kw)


def test_ingest_stores_everything(store):
    total = sum(len(chunk_document(d)) for d in DOCS)
    rep = run(store)
    assert rep.docs == 10 and rep.chunks == total and rep.embedded == total
    st = store.stats()
    assert st["documents"] == 10 and st["chunks"] == total
    assert st["by_status"]["internal/current"] >= 1 and st["by_status"]["public/deprecated"] >= 1


def test_second_run_embeds_nothing(store):
    run(store)
    emb = FakeEmbedder()
    rep = run(store, emb=emb)
    assert rep.embedded == 0 and rep.unchanged == rep.chunks and emb.calls == 0


def test_changed_text_re_embeds_only_that_chunk(store):
    run(store)
    changed = [replace(d, body=d.body.replace("within 7 days", "within 10 days")) if d.doc_id == "refund-policy-v2" else d
               for d in DOCS]
    emb = FakeEmbedder()
    rep = run(store, docs=changed, emb=emb)
    assert 1 <= rep.embedded < 5 and emb.calls == rep.embedded


def test_status_change_updates_metadata_without_re_embedding(store):
    run(store)
    changed = [replace(d, meta={**d.meta, "status": "deprecated"}) if d.doc_id == "pricing-plans" else d for d in DOCS]
    emb = FakeEmbedder()
    run(store, docs=changed, emb=emb)
    assert emb.calls == 0
    ids = {h.doc_id for h in store.keyword_search("plans pricing users invoices", 20)}
    assert "pricing-plans" not in ids  # now deprecated -> filtered out by default


def test_deprecated_and_internal_are_filtered_before_ranking(store):
    run(store)
    default = store.keyword_search("30-day money-back guarantee refund", 10)
    assert default and all(h.status == "current" and h.audience == "public" for h in default)
    assert "refund-policy-v1" not in {h.doc_id for h in default}
    allhits = store.keyword_search("30-day money-back guarantee refund", 10, ("public", "internal"), ("current", "deprecated"))
    assert "refund-policy-v1" in {h.doc_id for h in allhits}


def test_internal_runbook_never_leaks_by_default(store):
    run(store)
    q = "goodwill refund exception support agents approve"
    assert "internal-escalation-runbook" not in {h.doc_id for h in store.keyword_search(q, 10)}
    assert "internal-escalation-runbook" in {h.doc_id for h in store.keyword_search(q, 10, ("public", "internal"), ("current",))}
    emb = FakeEmbedder()
    leak = [h for h in store.vector_search(emb.embed_query(q), 50) if h.audience == "internal"]
    assert leak == []


def test_vector_search_finds_exact_chunk_and_respects_filters(store):
    run(store)
    target = next(c for c in chunk_document(next(d for d in DOCS if d.doc_id == "pricing-plans")))
    hits = store.vector_search(FakeEmbedder().embed_query(target.text), 3)
    assert hits[0].chunk_id == target.chunk_id and hits[0].score > 0.99
    assert all(h.status == "current" and h.audience == "public" for h in hits)


def test_natural_language_question_still_reaches_the_right_chunk(store):
    """Keyword-only ranking is noisy for full questions ("card" also matches failed-payment docs),
    which is exactly why A1.3 fuses it with vector search. Recall in the top 3 is what we require here."""
    run(store)
    hits = store.keyword_search("How long does a refund take to reach my card?", 3)
    assert "refund-policy-v2:4" in [h.chunk_id for h in hits]  # "How refunds are paid": 5 to 10 business days


def test_prune_removes_missing_documents(store):
    run(store)
    rep = run(store, docs=[d for d in DOCS if d.doc_id != "webhooks"], prune=True)
    assert rep.deleted_docs == 1
    assert "webhooks" not in {h.doc_id for h in store.keyword_search("webhook signature HMAC", 10)}


def test_embedding_model_change_is_detected(store):
    with pytest.raises(SchemaMismatch):
        store.init_schema(16, "fake-8")
    with pytest.raises(SchemaMismatch):
        store.init_schema(DIM, "other-model")


def test_gemini_sized_vectors_use_halfvec_and_hnsw():
    """3072-dim embeddings exceed pgvector's 2000-dim index limit for `vector`, so halfvec must work end to end."""
    class Big(FakeEmbedder):
        model = "fake-3072"

        def embed(self, texts):
            self.calls += len(texts)
            out = []
            for t in texts:
                v = np.random.RandomState(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).randn(3072)
                out.append((v / np.linalg.norm(v)).tolist())
            return out

    st = Store(S.database_url, f"docqa_test_{uuid.uuid4().hex[:8]}")
    try:
        st.init_schema(3072, "fake-3072")
        col = st.conn.execute(
            "SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'").fetchone()["t"]
        assert col == "halfvec(3072)"
        emb = Big()
        ingest(DOCS, emb, st)
        target = next(iter(chunk_document(next(d for d in DOCS if d.doc_id == "api-rate-limits"))))
        hits = st.vector_search(emb.embed_query(target.text), 3)
        assert hits[0].chunk_id == target.chunk_id
    finally:
        st.drop_schema()
        st.close()
