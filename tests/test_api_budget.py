"""Persistent daily budget (A1.7b): survives restarts, fails closed, switched on by API_BUDGET_STORE=db."""
from __future__ import annotations

import psycopg
import pytest

from docqa.api import ChatService, DailyBudget, PersistentBudget, get_runtime
from docqa.config import load_settings
from docqa.retrieval import Retriever
from docqa.store import Store

SCHEMA = "docqa_test_budget"


class Boom:
    def usage_today(self) -> int:
        raise RuntimeError("database down")

    def add_usage(self, n: int) -> int:
        raise RuntimeError("database down")


class Counter:
    def __init__(self) -> None:
        self.used = 0

    def usage_today(self) -> int:
        return self.used

    def add_usage(self, n: int) -> int:
        self.used += n
        return self.used


def test_persistent_budget_blocks_at_the_limit():
    svc = Counter()
    b = PersistentBudget(svc, 3)
    assert not b.exhausted()
    b.add(2)
    assert not b.exhausted()
    b.add(1)
    assert b.exhausted()


def test_persistent_budget_ignores_zero_calls():
    svc = Counter()
    PersistentBudget(svc, 3).add(0)
    assert svc.used == 0


def test_persistent_budget_fails_closed_when_the_database_is_unreadable_and_never_raises_on_add():
    b = PersistentBudget(Boom(), 3)
    assert b.exhausted() is True
    b.add(1)  # must not raise: the visitor already got an answer


def test_api_budget_store_env_selects_the_implementation(monkeypatch):
    get_runtime.cache_clear()
    monkeypatch.setenv("API_BUDGET_STORE", "db")
    assert isinstance(get_runtime().budget, PersistentBudget)
    get_runtime.cache_clear()
    monkeypatch.setenv("API_BUDGET_STORE", "memory")
    assert isinstance(get_runtime().budget, DailyBudget)
    get_runtime.cache_clear()


def _service() -> ChatService:
    s = load_settings()

    def build():
        store = Store(s.database_url, SCHEMA)
        return store, Retriever(store)

    return ChatService(s, build=build)


@pytest.fixture()
def clean_schema():
    s = load_settings()
    try:
        st = Store(s.database_url, SCHEMA)
    except psycopg.OperationalError:
        pytest.skip("Postgres is not running")
    st.drop_schema()
    st.close()
    yield
    st = Store(s.database_url, SCHEMA)
    st.drop_schema()
    st.close()


def test_usage_is_counted_in_the_database_and_survives_a_restart(clean_schema):
    first = _service()
    assert first.usage_today() == 0
    assert first.add_usage(2) == 2
    assert first.add_usage(1) == 3

    restarted = _service()  # a new process after a Render spin-down: nothing is kept in memory
    assert restarted.usage_today() == 3
    budget = PersistentBudget(restarted, 4)
    assert not budget.exhausted()
    budget.add(1)
    assert budget.exhausted()
