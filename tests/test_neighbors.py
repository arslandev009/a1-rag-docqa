"""Neighbour expansion (A1.6b): ordering, dedupe, opt-in, and above all: neighbours cannot cross the access filter."""
from __future__ import annotations

import json
from types import SimpleNamespace

import psycopg
import pytest

from docqa.chunker import Chunk
from docqa.config import load_settings
from docqa.generate import Assistant
from docqa.retrieval import RetrievalResult, Retrieved, Retriever
from docqa.store import Hit, Store


def hit(cid: str, doc: str = "d1", content: str = "text") -> Hit:
    return Hit(cid, doc, "Title", f"{doc} > {cid}", content, "https://x.test", "current", "public", 0.5)


class FakeStore:
    def __init__(self, neighbours: list[Hit]) -> None:
        self.neighbours, self.calls = neighbours, []

    def neighbors(self, chunk_ids, radius, audiences, statuses):
        self.calls.append((list(chunk_ids), radius, tuple(audiences), tuple(statuses)))
        return self.neighbours


def items(*ids: str) -> list[Retrieved]:
    return [Retrieved(hit(i), 1.0 / n, vector_rank=n) for n, i in enumerate(ids, 1)]


# ------------------------------------------------------------------------------ Retriever.expand
def test_expand_is_off_by_default_and_does_not_touch_the_store(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_NEIGHBORS", raising=False)
    st = FakeStore([hit("n1")])
    r = Retriever(st)
    assert r.neighbors == 0
    base = items("a", "b")
    assert r.expand(base) == base
    assert st.calls == []


def test_env_switches_it_on(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_NEIGHBORS", "1")
    monkeypatch.setenv("RETRIEVAL_NEIGHBOR_TOP", "2")
    r = Retriever(FakeStore([]))
    assert (r.neighbors, r.neighbor_top) == (1, 2)


def test_expand_keeps_originals_first_dedupes_and_flags_neighbours():
    st = FakeStore([hit("a"), hit("n1"), hit("n1"), hit("b"), hit("n2")])  # SQL may repeat or include the anchors
    out = Retriever(st, neighbors=1, neighbor_top=2).expand(items("a", "b", "c"))
    assert [r.chunk_id for r in out] == ["a", "b", "c", "n1", "n2"]
    assert [r.is_neighbor for r in out] == [False, False, False, True, True]
    assert st.calls[0][0] == ["a", "b"]  # only the best neighbor_top hits are expanded
    assert st.calls[0][1] == 1


def test_expand_passes_the_access_filter_through():
    st = FakeStore([])
    Retriever(st, neighbors=1).expand(items("a"))
    assert st.calls[0][2:] == (("public",), ("current",))


# --------------------------------------------------------------------------------- Assistant
class FakeRetriever:
    def __init__(self, with_expand: bool) -> None:
        self.base = items("a", "b")
        if with_expand:
            self.expand = lambda found: [*found, Retrieved(hit("n1", content="PRICE IS 19 USD"), 0.0, is_neighbor=True)]

    def retrieve(self, question, *, k, mode):
        return RetrievalResult(self.base, 0.9, None)


class FakeLLM:
    provider, model = "fake", "m"

    def __init__(self, cite: str) -> None:
        self.cite, self.prompts = cite, []

    def chat(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        text = json.dumps({"answer": f"It costs 19 USD [{self.cite}].", "citations": [self.cite], "abstained": False, "reason": ""})
        return SimpleNamespace(text=text, prompt_tokens=1, completion_tokens=1, cost_usd=0.0)


def test_assistant_shows_neighbours_to_the_model_and_can_cite_them():
    llm = FakeLLM("S3")
    res = Assistant(FakeRetriever(True), llm, min_cosine=0.5).ask("How much is the plan?")
    assert "PRICE IS 19 USD" in llm.prompts[0]
    assert [c.chunk_id for c in res.citations] == ["n1"]
    assert res.retrieved == ["a", "b", "n1"] and not res.abstained


def test_assistant_still_works_with_a_retriever_that_has_no_expand():
    llm = FakeLLM("S1")
    res = Assistant(FakeRetriever(False), llm, min_cosine=0.5).ask("How much is the plan?")
    assert res.retrieved == ["a", "b"] and res.citations[0].chunk_id == "a"


# ---------------------------------------------------------------- real SQL (skipped without Postgres)
SCHEMA = "docqa_test_neighbors"


@pytest.fixture()
def store():
    s = load_settings()
    try:
        st = Store(s.database_url, SCHEMA)
    except psycopg.OperationalError:
        pytest.skip("Postgres is not running")
    st.init_schema(4, "fake-model", reset=True)
    for doc_id, status, audience in (("pub", "current", "public"), ("secret", "current", "internal"),
                                     ("old", "deprecated", "public")):
        meta = {"status": status, "audience": audience, "version": "1", "source_url": f"https://x.test/{doc_id}"}
        st.upsert_document(SimpleNamespace(doc_id=doc_id, title=doc_id.title(), meta=meta))
        for i in range(3):
            st.upsert_chunk(Chunk(f"{doc_id}#{i}", doc_id, i, f"{doc_id} > s{i}", f"text {doc_id} {i}", meta),
                            [0.1, 0.2, 0.3, 0.4], "h")
    yield st
    st.drop_schema()
    st.close()


def ids(hits) -> list[str]:
    return [h.chunk_id for h in hits]


def test_sql_neighbours_stay_in_the_document_and_in_order(store):
    assert ids(store.neighbors(["pub#1"], 1)) == ["pub#0", "pub#1", "pub#2"]
    assert ids(store.neighbors(["pub#0"], 1)) == ["pub#0", "pub#1"]
    assert store.neighbors(["pub#0"], 0) == []
    assert store.neighbors([], 1) == []


def test_sql_neighbours_never_cross_the_access_filter(store):
    assert store.neighbors(["secret#0", "secret#1", "secret#2"], 2) == []  # internal doc
    assert store.neighbors(["old#1"], 1) == []  # deprecated doc
    # the debug flags reach them, which proves the filter (not their absence) is what hides them
    assert ids(store.neighbors(["secret#1"], 1, audiences=("public", "internal"))) == ["secret#0", "secret#1", "secret#2"]


def test_sql_neighbours_of_two_anchors_may_repeat_a_chunk_but_expand_dedupes(store):
    got = ids(store.neighbors(["pub#0", "pub#2"], 1))
    assert set(got) == {"pub#0", "pub#1", "pub#2"}
    anchors = [Retrieved(h, 1.0) for h in store.neighbors(["pub#0", "pub#2"], 1) if h.chunk_id in ("pub#0", "pub#2")]
    out = Retriever(store, neighbors=1, neighbor_top=2).expand(anchors)
    assert [r.chunk_id for r in out] == ["pub#0", "pub#2", "pub#1"]  # anchors first, the shared neighbour once
