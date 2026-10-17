import json

import pytest

from docqa.chunker import chunk_document, pack, split_sections
from docqa.config import load_settings
from docqa.loader import load_corpus, parse_document

S = load_settings()
DOCS = load_corpus(S.corpus_dir)
IDS = {d.doc_id for d in DOCS}


def test_corpus_loads_with_valid_metadata():
    assert len(DOCS) == 10
    by_id = {d.doc_id: d for d in DOCS}
    assert by_id["refund-policy-v1"].meta["status"] == "deprecated"
    assert by_id["refund-policy-v1"].meta["superseded_by"] == "refund-policy-v2"
    assert by_id["internal-escalation-runbook"].meta["audience"] == "internal"


def test_missing_front_matter_is_rejected():
    with pytest.raises(ValueError):
        parse_document("x", "# no front matter")


def test_chunk_ids_unique_and_deterministic():
    a = [c.chunk_id for d in DOCS for c in chunk_document(d)]
    b = [c.chunk_id for d in DOCS for c in chunk_document(d)]
    assert a == b and len(a) == len(set(a))


def test_chunks_keep_heading_and_metadata():
    for d in DOCS:
        for c in chunk_document(d):
            assert c.text.startswith(c.heading_path)
            assert c.meta["status"] in {"current", "deprecated"}
            assert c.meta["doc_id"] == d.doc_id and c.meta["source_url"]


def test_chunk_size_bounded():
    for d in DOCS:
        for c in chunk_document(d, max_chars=400):
            body = c.text[len(c.heading_path):].strip()
            assert 0 < len(body) <= 400 * 1.6  # overlap sentence may exceed slightly


def test_long_paragraph_is_split_on_sentences():
    para = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = pack(para, 300, overlap_sentences=1)
    assert len(chunks) > 3 and all(len(c) <= 300 * 1.6 for c in chunks)


def test_heading_paths():
    paths = [p for p, _ in split_sections("# T\n\n## A\n\nx\n\n### B\n\ny\n\n## C\n\nz", "T")]
    assert paths == ["T > A", "T > A > B", "T > C"]


def test_eval_set_is_consistent_with_corpus():
    types = {"factual", "numeric", "multi_doc", "version_conflict", "negation", "ambiguity",
             "unanswerable", "access_control", "injection", "paraphrase"}
    rows = [json.loads(line) for line in S.eval_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 30 and len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["type"] in types
        assert set(r["sources"]) <= IDS
        if r["type"] in {"unanswerable", "injection"}:
            assert r["sources"] == []
