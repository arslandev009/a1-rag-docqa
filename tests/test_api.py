"""API tests: no database, no network, no LLM. A fake service stands in for ChatService."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from docqa.api import ApiConfig, DailyBudget, RateLimiter, Runtime, get_runtime, router
from docqa.generate import AnswerResult, Citation


def _result(**over) -> AnswerResult:
    base = dict(
        question="q", answer="Refunds take 5-10 days [S1].", abstained=False, reason="ok",
        citations=[Citation("S1", "refund-policy-v2#timing", "refund-policy-v2", "Refund policy", "Timing", "https://x.test/r")],
        provider="fake", model="m1", llm_calls=1, latency_s=0.5, cost_usd=0.0001,
        context=["SECRET CONTEXT TEXT"], error="boom internal detail",
    )
    base.update(over)
    return AnswerResult(**base)


class FakeService:
    def __init__(self, ok: bool = True, result: AnswerResult | None = None, boom: bool = False) -> None:
        self.ok, self.result, self.boom, self.calls = ok, result or _result(), boom, []

    def ping(self) -> bool:
        return self.ok

    def ask(self, provider: str, question: str) -> AnswerResult:
        if self.boom:
            raise RuntimeError("db exploded with password=hunter2")
        self.calls.append((provider, question))
        return self.result

    def sources(self) -> list[dict]:
        return [{"doc_id": "refund-policy-v2", "title": "Refund policy", "source_url": "https://x.test/r",
                 "version": "2", "effective_date": "2025-01-01"}]


def build(service=None, *, rate: int = 100, budget: int = 1000, **cfg):
    service = service or FakeService()
    rt = Runtime(service, ApiConfig(rate_per_minute=rate, daily_llm_calls=budget, **cfg), RateLimiter(rate), DailyBudget(budget))
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_runtime] = lambda: rt
    return TestClient(app), service, rt


Q = {"question": "How long does a refund take?"}


def test_chat_returns_answer_and_citations_without_internals():
    client, svc, _ = build()
    r = client.post("/chat", json=Q)
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("Refunds take")
    assert body["citations"][0]["doc_id"] == "refund-policy-v2"
    assert body["abstained"] is False
    assert "SECRET CONTEXT" not in r.text and "boom internal" not in r.text  # context and error detail never leave
    assert svc.calls == [("gemini", Q["question"])]


def test_abstention_is_a_normal_200():
    res = _result(answer="I couldn't find this.", abstained=True, reason="not_in_sources", citations=[], llm_calls=1)
    client, _, _ = build(FakeService(result=res))
    r = client.post("/chat", json=Q)
    assert r.status_code == 200 and r.json()["abstained"] is True and r.json()["citations"] == []


def test_provider_error_becomes_502_without_details():
    client, _, _ = build(FakeService(result=_result(reason="llm_error", llm_calls=0)))
    r = client.post("/chat", json=Q)
    assert r.status_code == 502 and "boom" not in r.text


def test_unexpected_exception_becomes_503_and_hides_the_message():
    client, _, _ = build(FakeService(boom=True))
    r = client.post("/chat", json=Q)
    assert r.status_code == 503 and "hunter2" not in r.text


def test_provider_allow_list():
    client, svc, _ = build()
    assert client.post("/chat", json={**Q, "provider": "openai"}).status_code == 400
    assert client.post("/chat", json={**Q, "provider": "GROQ"}).status_code == 200
    assert svc.calls[-1][0] == "groq"


def test_question_validation():
    client, _, _ = build()
    assert client.post("/chat", json={"question": "  "}).status_code == 422
    assert client.post("/chat", json={"question": "x" * 501}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422


def test_gateway_secret_is_enforced_but_health_stays_open():
    client, _, _ = build(gateway_secret="s3cret")
    assert client.post("/chat", json=Q).status_code == 401
    assert client.post("/chat", json=Q, headers={"X-Gateway-Secret": "wrong"}).status_code == 401
    assert client.get("/sources").status_code == 401
    assert client.post("/chat", json=Q, headers={"X-Gateway-Secret": "s3cret"}).status_code == 200
    assert client.get("/sources", headers={"X-Gateway-Secret": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200


def test_rate_limit_is_per_client_ip():
    client, _, _ = build(rate=2)
    codes = [client.post("/chat", json=Q).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    other = client.post("/chat", json=Q, headers={"X-Real-Client-IP": "203.0.113.9"})
    assert other.status_code == 200


def test_daily_budget_kill_switch():
    client, _, rt = build(budget=2)
    assert client.post("/chat", json=Q).status_code == 200  # llm_calls=1
    assert client.post("/chat", json=Q).status_code == 200  # llm_calls=1 -> budget used up
    assert rt.budget.exhausted() is True
    assert client.post("/chat", json=Q).status_code == 503


def test_health_reports_not_ready():
    client, _, _ = build(FakeService(ok=False))
    assert client.get("/health").status_code == 503


def test_sources_lists_public_documents():
    client, _, _ = build()
    r = client.get("/sources")
    assert r.status_code == 200 and r.json()[0]["doc_id"] == "refund-policy-v2"


def test_rate_limiter_window_slides():
    now = [0.0]
    rl = RateLimiter(2, clock=lambda: now[0])
    assert rl.allow("a") and rl.allow("a") and not rl.allow("a")
    now[0] = 61.0
    assert rl.allow("a")


def test_daily_budget_rolls_over_at_midnight_utc():
    day = ["2026-10-17"]
    b = DailyBudget(1, today=lambda: day[0])
    b.add(1)
    assert b.exhausted()
    day[0] = "2026-10-18"
    assert not b.exhausted()
