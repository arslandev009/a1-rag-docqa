"""Error mapping (A1.7c): a failure must say what kind it is, without leaking internals."""
from __future__ import annotations

import logging

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from docqa.api import ApiConfig, DailyBudget, RateLimiter, Runtime, classify_error, get_runtime, router


def exc_from(module: str, name: str, message: str = "secret detail 10.0.0.5") -> Exception:
    return type(name, (Exception,), {"__module__": module})(message)


class Raising:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def ping(self) -> bool:
        return True

    def ask(self, provider: str, question: str):
        raise self.exc

    def sources(self) -> list:
        return []


def client_for(exc: Exception) -> TestClient:
    rt = Runtime(Raising(exc), ApiConfig(rate_per_minute=100, daily_llm_calls=1000), RateLimiter(100), DailyBudget(1000))
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_runtime] = lambda: rt
    return TestClient(app)


@pytest.mark.parametrize("module,name,expected", [
    ("openai", "RateLimitError", "provider"),
    ("openai._exceptions", "APITimeoutError", "provider"),
    ("httpx", "ConnectError", "provider"),
    ("llm_client.client", "ProviderError", "provider"),
    ("psycopg.errors", "AdminShutdown", "database"),
    ("builtins", "RuntimeError", "other"),
])
def test_classify_error(module, name, expected):
    assert classify_error(exc_from(module, name)) == expected


def test_real_exception_classes_are_classified():
    assert classify_error(psycopg.OperationalError("server closed the connection")) == "database"
    assert classify_error(ValueError("x")) == "other"


def test_provider_failure_is_a_502_with_a_helpful_message_and_no_internals(caplog):
    c = client_for(exc_from("openai", "RateLimitError"))
    with caplog.at_level(logging.ERROR, logger="docqa.api"):
        r = c.post("/chat", json={"question": "How long does a refund take?"})
    assert r.status_code == 502 and "provider is busy" in r.json()["detail"]
    assert "10.0.0.5" not in r.text
    assert "RateLimitError" in caplog.text and "kind=provider" in caplog.text  # the log says what failed


def test_database_failure_is_a_503_that_says_so():
    r = client_for(exc_from("psycopg", "OperationalError")).post("/chat", json={"question": "How long does a refund take?"})
    assert r.status_code == 503 and "database is waking up" in r.json()["detail"] and "10.0.0.5" not in r.text


def test_unknown_failure_stays_generic():
    r = client_for(RuntimeError("password=hunter2")).post("/chat", json={"question": "How long does a refund take?"})
    assert r.status_code == 503 and r.json()["detail"] == "The assistant is temporarily unavailable." and "hunter2" not in r.text
