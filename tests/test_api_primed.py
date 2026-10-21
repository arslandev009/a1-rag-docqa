"""A rebuilt Store must be primed (A1.7d): the demo broke when the connection was re-opened and nobody told the new
Store the vector size, so every search failed until /health happened to run."""
from __future__ import annotations

from types import SimpleNamespace

import psycopg
import pytest

from docqa.api import ChatService, prime_store
from docqa.chunker import Chunk
from docqa.config import load_settings
from docqa.retrieval import Retriever
from docqa.store import Store

SCHEMA = "docqa_test_primed"
VECS = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


@pytest.fixture()
def database():
    s = load_settings()
    try:
        st = Store(s.database_url, SCHEMA)
    except psycopg.OperationalError:
        pytest.skip("Postgres is not running")
    st.init_schema(4, "fake-model", reset=True)
    meta = {"status": "current", "audience": "public", "version": "1", "source_url": "https://x.test/plans"}
    st.upsert_document(SimpleNamespace(doc_id="plans", title="Plans", meta=meta))
    for i, v in enumerate(VECS):
        st.upsert_chunk(Chunk(f"plans:{i}", "plans", i, f"Plans > s{i}", f"section {i}", meta), v, "h")
    st.close()
    yield s
    st = Store(s.database_url, SCHEMA)
    st.drop_schema()
    st.close()


def test_a_fresh_store_cannot_search_until_it_is_primed(database):
    fresh = Store(database.database_url, SCHEMA)
    with pytest.raises(AssertionError):  # the trap that broke the live demo
        fresh.vector_search([0, 0, 0, 1], 1)
    prime_store(fresh)
    assert fresh.vector_search([0, 0, 0, 1], 1)[0].chunk_id == "plans:3"
    fresh.close()


def test_prime_store_is_harmless_on_an_empty_database():
    s = load_settings()
    try:
        st = Store(s.database_url, "docqa_test_primed_empty")
    except psycopg.OperationalError:
        pytest.skip("Postgres is not running")
    try:
        prime_store(st)  # nothing ingested yet: must not raise
    finally:
        st.drop_schema()
        st.close()


def test_service_recovers_after_the_database_drops_the_connection(database):
    def build():
        st = Store(database.database_url, SCHEMA)
        return st, Retriever(st)

    svc = ChatService(database, build=build)
    assert svc.ping()  # first build
    svc.store.conn.close()  # what Neon does to an idle connection
    hits = svc._call(lambda: svc.store.vector_search([0, 0, 0, 1], 1))  # rebuild, then search: no ping in between
    assert hits and hits[0].chunk_id == "plans:3"
    assert len(svc.sources()) == 1


def test_a_search_as_the_very_first_request_works_without_a_ping(database):
    def build():
        st = Store(database.database_url, SCHEMA)
        return st, Retriever(st)

    svc = ChatService(database, build=build)  # nobody has called ping() or health
    assert svc._call(lambda: svc.store.vector_search([1, 0, 0, 0], 1))[0].chunk_id == "plans:0"
