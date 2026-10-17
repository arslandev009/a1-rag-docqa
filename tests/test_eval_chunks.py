"""Chunk-level eval (A1.6c): the labelling heuristic, and the grid on real SQL (skipped without Postgres)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import psycopg
import pytest

from docqa.chunker import Chunk
from docqa.config import load_settings
from docqa.eval_chunks import auto_label, grid, key_tokens, labels_for
from docqa.store import Store

# the real pricing-plans chunks (shortened), as printed from the database
PRICING = [
    (0, "Ledgerly Plans and Pricing\n\nLedgerly is invoicing and payments software for small businesses. All prices are in USD per month."),
    (1, "Ledgerly Plans and Pricing > Plans\n\n- **Starter, $19/month:** 1 user, up to 50 invoices per month, 1 integration. "
        "- **Growth, $49/month:** 5 users, up to 500 invoices per month."),
    (2, "Ledgerly Plans and Pricing > Extra users\n\nGrowth accounts can add users for $6 per user per month, up to 15 users."),
    (3, "Ledgerly Plans and Pricing > Billing options\n\nAnnual billing gives two months free: you pay for 10 months and use 12."),
    (4, "Ledgerly Plans and Pricing > Free trial\n\nStarter and Growth include a 14-day free trial and no credit card is required."),
]


def chunks(doc: str = "pricing-plans", rows=PRICING) -> list[dict]:
    return [{"chunk_id": f"{doc}:{i}", "doc_id": doc, "chunk_index": i, "heading_path": "x", "content": t} for i, t in rows]


# ----------------------------------------------------------------------------------------- labelling
def test_key_tokens_weigh_numbers_more_than_words():
    t = key_tokens("Starter, $19/month, up to 50 invoices per month")
    assert t["$19"] == 2.0 and t["50"] == 2.0 and t["starter"] == 1.0 and "invoices" in t


def test_q44_gold_chunk_is_the_plans_chunk():
    assert auto_label("Starter, $19/month, up to 50 invoices per month", chunks()) == ["pricing-plans:1"]


def test_q48_gold_chunk_is_the_free_trial_chunk():
    assert auto_label("14-day trial, no card for Starter/Growth", chunks()) == ["pricing-plans:4"]


def test_numbers_match_whole_numbers_only():
    only_500 = chunks(rows=[(0, "up to 500 invoices"), (1, "up to 50 invoices")])
    assert auto_label("50 invoices", only_500) == ["pricing-plans:1"]


def test_every_gold_document_contributes_its_best_chunk():
    both = chunks("a", [(0, "refund within 7 days"), (1, "unrelated")]) + chunks("b", [(0, "nothing"), (1, "trial of 14 days")])
    assert auto_label("7 days refund and 14 days trial", both) == ["a:0", "b:1"]


def test_no_overlap_means_no_label():
    assert auto_label("zebra quartz", chunks()) == []


# ----------------------------------------------------------------------- real SQL: the q44-shaped case
SCHEMA = "docqa_test_chunk_eval"
VECS = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1]]


@pytest.fixture()
def store():
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
    yield st
    st.drop_schema()
    st.close()


def test_grid_shows_that_a_wider_window_recovers_a_gold_chunk_that_is_not_adjacent(store):
    # the query vector is closest to chunk 3; the gold chunk is chunk 1 (two steps away, like pricing-plans:1 for q44)
    q = {"id": "q1", "question": "zzz", "answerable": True, "type": "paraphrase", "expected": "x", "sources": ["plans"]}
    labels = {"q1": {"gold_chunks": ["plans:1"]}}
    rows = grid(store, None, [q], labels, {"zzz": [0, 0, 0, 1]}, weights=(0.0,), candidate_ks=(5,), ks=(1,), radii=(0, 1, 2))
    by_radius = {r["neighbors"]: r for r in rows}
    assert by_radius[0]["ctx_all"] == 0.0 and by_radius[0]["avg_chunks"] == 1
    assert by_radius[1]["ctx_all"] == 0.0 and by_radius[1]["avg_chunks"] == 3  # chunks 2, 3, 4: still misses chunk 1
    assert by_radius[2]["ctx_all"] == 1.0 and by_radius[2]["avg_chunks"] == 4  # chunks 1..4
    assert all(r["violations"] == 0 for r in rows)


def test_labels_are_written_once_and_reused(store, tmp_path):
    q = {"id": "q1", "question": "zzz", "answerable": True, "type": "factual", "expected": "section 1", "sources": ["plans"]}
    path = tmp_path / "gold_chunks.json"
    first = labels_for(path, store, [q])
    assert first["q1"]["gold_chunks"] == ["plans:1"] and first["q1"]["reviewed"] is False
    data = json.loads(path.read_text(encoding="utf-8"))
    data["labels"]["q1"]["gold_chunks"] = ["plans:3"]  # a human correction must survive
    path.write_text(json.dumps(data), encoding="utf-8")
    assert labels_for(path, store, [q])["q1"]["gold_chunks"] == ["plans:3"]
    assert labels_for(path, store, [q], relabel=True)["q1"]["gold_chunks"] == ["plans:1"]
