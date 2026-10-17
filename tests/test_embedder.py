from types import SimpleNamespace

try:  # newer openai releases ship on httpx2
    import httpx2 as httpx
except ImportError:  # older ones use httpx
    import httpx
import numpy as np
import pytest
from openai import RateLimitError

from docqa.embedder import Embedder


def _rate_limit() -> RateLimitError:
    resp = httpx.Response(429, request=httpx.Request("POST", "http://test"))
    return RateLimitError("slow down", response=resp, body=None)


class FakeClient:
    def __init__(self, fail_times: int = 0, exc: Exception | None = None):
        self.calls: list[list[str]] = []
        self.fail, self.exc = fail_times, exc
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model, input):
        self.calls.append(list(input))
        if self.fail > 0:
            self.fail -= 1
            raise self.exc
        data = [SimpleNamespace(index=i, embedding=[3.0 * (i + 1), 4.0 * (i + 1), 0.0]) for i in range(len(input))]
        return SimpleNamespace(data=list(reversed(data)))  # API order is not guaranteed; we sort by index


def make(client, **kw):
    return Embedder("local", "fake-model", _client=client, **kw)


def test_batches_and_preserves_order():
    client = FakeClient()
    vecs = make(client, batch_size=2).embed(["a", "b", "c", "d", "e"])
    assert [len(c) for c in client.calls] == [2, 2, 1]
    assert len(vecs) == 5
    # within each batch item 0 is [3,4,0]/5 and item 1 is [6,8,0]/10: both normalise to the same direction
    assert np.allclose(vecs[0], [0.6, 0.8, 0.0])


def test_vectors_are_unit_length():
    for v in make(FakeClient()).embed(["x", "y", "z"]):
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_empty_input_makes_no_calls():
    client = FakeClient()
    assert make(client).embed([]) == [] and client.calls == []


def test_retries_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    client = FakeClient(fail_times=2, exc=_rate_limit())
    assert len(make(client, max_retries=4).embed(["a"])) == 1
    assert len(client.calls) == 3


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    client = FakeClient(fail_times=10, exc=_rate_limit())
    with pytest.raises(RateLimitError):
        make(client, max_retries=2).embed(["a"])
    assert len(client.calls) == 3


def test_non_retryable_error_is_raised_immediately():
    client = FakeClient(fail_times=1, exc=ValueError("bad request"))
    with pytest.raises(ValueError):
        make(client).embed(["a"])
    assert len(client.calls) == 1


def test_empty_model_is_rejected():
    with pytest.raises(RuntimeError):
        Embedder("gemini", "")


def test_missing_index_keeps_api_order():
    """Gemini's compatible endpoint returns index=None on each item; order must then be kept as returned."""
    items = [SimpleNamespace(index=None, embedding=[1.0, 0.0, 0.0]), SimpleNamespace(index=None, embedding=[0.0, 1.0, 0.0])]
    client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda model, input: SimpleNamespace(data=items)))
    vecs = make(client).embed(["a", "b"])
    assert np.allclose(vecs[0], [1, 0, 0]) and np.allclose(vecs[1], [0, 1, 0])
