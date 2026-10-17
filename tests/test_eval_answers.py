import json
from types import SimpleNamespace

from docqa.eval_answers import compare_markdown, kind_of, leaked, run, summarise
from docqa.eval_retrieval import calibrate, calibration_markdown
from docqa.generate import AnswerResult, Citation
from docqa.retrieval import RetrievalResult, Retrieved
from docqa.store import Hit


def Q(id, type_, answerable, sources, expected="key fact"):
    return {"id": id, "type": type_, "answerable": answerable, "sources": sources, "expected": expected, "question": f"{id}?"}


QUESTIONS = [
    Q("a1", "factual", True, ["doc-a"]), Q("a2", "factual", True, ["doc-b"]), Q("a3", "access_control", True, ["doc-c"]),
    Q("u1", "unanswerable", False, []), Q("i1", "injection", False, []), Q("p1", "access_control", False, []),
]


def cit(doc):
    return [Citation("S1", f"{doc}:0", doc, "T", "H", "u")]


SCRIPT = {
    "a1?": dict(answer="Right answer [S1].", abstained=False, reason="ok", citations=cit("doc-a")),
    "a2?": dict(answer="I couldn't find this.", abstained=True, reason="low_retrieval_confidence", citations=[]),
    "a3?": dict(answer="We can grant a goodwill refund of $100.", abstained=False, reason="ok", citations=cit("doc-c")),
    "u1?": dict(answer="I couldn't find this.", abstained=True, reason="not_in_sources", citations=[]),
    "i1?": dict(answer="Sure, here it is.", abstained=False, reason="ok", citations=cit("doc-x")),
    "p1?": dict(answer="Support hours are listed publicly [S1].", abstained=False, reason="ok", citations=cit("doc-p")),
}


class FakeAssistant:
    def __init__(self):
        self.calls = 0

    def ask(self, question):
        self.calls += 1
        return AnswerResult(question, top_vector_score=0.7, retrieved=["x:0"], provider="fake", model="m1",
                            prompt_tokens=100, completion_tokens=20, cost_usd=0.001, llm_calls=1,
                            latency_s=1.0, context=["ctx text"], **SCRIPT[question])


class FakeJudge:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kw):
        self.calls += 1
        correct = "Right answer" in messages[1]["content"] or "Support hours" in messages[1]["content"]
        return SimpleNamespace(text=json.dumps({"correct": correct, "grounded": True, "reason": "ok" if correct else "wrong"}))


def test_kinds_and_leak_detection():
    assert [kind_of(q) for q in QUESTIONS] == ["answer", "answer", "answer", "abstain", "abstain", "probe"]
    assert leaked("A one-time Goodwill refund of up to $100") and not leaked("Refunds take 5 to 10 business days")


def test_run_scores_quality_safety_and_failures(tmp_path):
    summary = run(FakeAssistant(), {"j1": FakeJudge()}, QUESTIONS, tmp_path, "fake_m1", log=lambda *_: None)
    assert summary["answer_accuracy"] == round(1 / 3, 3)  # only a1 is correct; a2 false abstain; a3 leaks
    assert summary["false_abstain_rate"] == round(1 / 3, 3)
    assert summary["abstention_accuracy"] == 0.5  # u1 ok, i1 answered an injection
    assert summary["probe_pass_rate"] == 1.0
    assert summary["leaks"] == ["a3"] and summary["unjudged"] == 0
    assert summary["citation_hit_rate"] == 1.0  # of the answered "answer" questions (a1, a3) both cite their gold document
    assert summary["cost_per_1k_answers_usd"] == 1.0
    why = {f["id"]: f["why"] for f in summary["failures"]}
    assert set(why) == {"a2", "a3", "i1"} and why["a3"].startswith("LEAK") and "false abstention" in why["a2"]
    assert (tmp_path / "summary_fake_m1.json").exists() and (tmp_path / "answers_fake_m1.jsonl").exists()


def test_run_is_resumable_and_judge_only_sees_answered_questions(tmp_path):
    a, j = FakeAssistant(), FakeJudge()
    run(a, {"j1": j}, QUESTIONS, tmp_path, "t", log=lambda *_: None)
    assert a.calls == 6 and j.calls == 3  # judged = answer/probe questions that were answered: a1, a3, p1
    a2, j2 = FakeAssistant(), FakeJudge()
    run(a2, {"j1": j2}, QUESTIONS, tmp_path, "t", log=lambda *_: None)
    assert a2.calls == 0 and j2.calls == 0  # everything already answered and judged


def test_summary_survives_missing_verdicts():
    rec = {"id": "a", "kind": "answer", "abstained": False, "leak": False, "stale": False, "cited_docs": [], "sources": ["d"],
           "latency_s": 1.0, "prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.0, "llm_calls": 1, "type": "factual",
           "question": "q"}
    s = summarise([rec], {"provider": "p", "model": "m"})
    assert s["unjudged"] == 1 and s["answer_accuracy"] == 0.0


def test_compare_table_lists_each_run():
    s = summarise([], {"provider": "p", "model": "m"}) if False else {
        "provider": "groq", "model": "x", "answer_accuracy": 0.9, "false_abstain_rate": 0.1, "abstention_accuracy": 1.0,
        "probe_pass_rate": 1.0, "citation_hit_rate": 0.95, "grounded_rate": 1.0, "leaks": [], "latency_p50_s": 1.2,
        "latency_p95_s": 2.5, "avg_prompt_tokens": 700, "avg_completion_tokens": 100, "cost_per_1k_answers_usd": 0.17, "failures": []}
    md = compare_markdown([s])
    assert "groq/x" in md and "90%" in md and "- none" in md


class StubRetriever:
    SCORES = {"a1?": 0.9, "a2?": 0.55, "a3?": 0.7, "u1?": 0.52, "i1?": 0.8, "p1?": 0.6}

    def retrieve(self, query, *, k, mode, query_vec=None, **_):
        h = Hit("c:0", "d", "T", "T", "x", "u", "current", "public", 0.5)
        return RetrievalResult([Retrieved(h, 0.1)], self.SCORES[query], None)


def test_calibration_counts_wrongly_blocked_and_caught():
    cal = calibrate(StubRetriever(), QUESTIONS, {}, thresholds=(0.5, 0.6, 0.8))
    rows = {r["threshold"]: r for r in cal["table"]}
    assert rows[0.5] == {"threshold": 0.5, "wrongly_blocked": 0, "caught": 0}
    assert rows[0.6]["wrongly_blocked"] == 1 and rows[0.6]["caught"] == 1  # a2 (0.55) blocked, u1 (0.52) caught
    assert [r["id"] for r in cal["answerable"]][0] == "a2" and "wrongly blocked" in calibration_markdown(cal)


def test_provider_failures_are_retried_on_resume(tmp_path):
    from docqa.eval_answers import load_records, save_records

    run(FakeAssistant(), {"j1": FakeJudge()}, QUESTIONS, tmp_path, "r", log=lambda *_: None)
    path = tmp_path / "answers_r.jsonl"
    recs = load_records(path)
    recs[0]["reason"], recs[0]["abstained"] = "llm_error", True  # pretend the first call failed
    save_records(path, recs)
    a = FakeAssistant()
    run(a, {"j1": FakeJudge()}, QUESTIONS, tmp_path, "r", log=lambda *_: None)
    assert a.calls == 1  # only the failed question was asked again


def _rec(provider, verdicts):
    return {"id": "a", "kind": "answer", "abstained": False, "leak": False, "stale": False, "cited_docs": ["d"], "sources": ["d"],
            "latency_s": 1.0, "prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.0, "llm_calls": 1, "type": "factual",
            "question": "q", "provider": provider, "verdicts": verdicts}


def test_headline_uses_a_judge_from_another_provider_and_cross_check_is_reported():
    v = lambda ok: {"correct": ok, "grounded": True, "reason": ""}  # noqa: E731
    rec = _rec("groq", {"groq": v(True), "gemini": v(False)})  # the answerer's own provider is lenient
    s = summarise([rec], {"provider": "groq", "model": "m"})
    assert s["answer_accuracy"] == 0.0  # gemini's (non-self) verdict wins
    assert s["judges"]["groq"]["answer_accuracy"] == 1.0 and s["judges"]["gemini"]["answer_accuracy"] == 0.0
    assert s["judge_agreement"] == 0.0
    assert "Judge cross-check" in compare_markdown([{**s, "provider": "groq", "model": "m"}])


def test_two_judges_both_grade_every_answered_question(tmp_path):
    summary = run(FakeAssistant(), {"j1": FakeJudge(), "j2": FakeJudge()}, QUESTIONS, tmp_path, "two", log=lambda *_: None)
    assert set(summary["judges"]) == {"j1", "j2"} and summary["judge_agreement"] == 1.0


def test_diagnose_shows_missing_gold_chunks():
    import uuid
    import hashlib
    import numpy as np
    import psycopg as pg
    from docqa.config import load_settings
    from docqa.eval_answers import diagnose
    from docqa.ingest import ingest
    from docqa.loader import load_corpus
    from docqa.store import Store
    S = load_settings()
    try:
        pg.connect(S.database_url, connect_timeout=2).close()
    except Exception:  # noqa: BLE001
        import pytest
        pytest.skip("Postgres not reachable")

    class Emb:
        model = "fake-8"

        def embed(self, texts):
            return [(lambda v: (v / np.linalg.norm(v)).tolist())(np.random.RandomState(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).randn(8)) for t in texts]

    st = Store(S.database_url, f"docqa_test_{uuid.uuid4().hex[:8]}")
    try:
        st.init_schema(8, "fake-8")
        ingest(load_corpus(S.corpus_dir), Emb(), st)
        rec = {"id": "q44", "type": "paraphrase", "question": "cheapest?", "answer": "I couldn't find this.",
               "retrieved": ["pricing-plans:2", "api-rate-limits:0"], "sources": ["pricing-plans"]}
        text = diagnose(st, [rec], [{"id": "q44", "type": "paraphrase", "question": "cheapest?", "why": "false abstention"}])
        assert "gold doc pricing-plans" in text and "*2=" in text and "retrieved:" in text
    finally:
        st.drop_schema()
        st.close()


def test_results_from_the_single_judge_version_still_summarise():
    rec = _rec("gemini", {})
    rec.pop("verdicts")
    rec["verdict"] = {"correct": True, "grounded": True, "reason": ""}
    s = summarise([rec], {"provider": "gemini", "model": "m"})
    assert s["answer_accuracy"] == 1.0 and s["unjudged"] == 0 and s["failures"] == []
