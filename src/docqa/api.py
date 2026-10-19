"""HTTP API for the Ledgerly assistant (A1.6).

Endpoints
  GET  /health   readiness (database reachable and ingested); open, for Docker / Render health checks
  GET  /sources  public documents the assistant is allowed to cite
  POST /chat     {"question": "...", "provider": "gemini"} -> answer + citations

Mount it in the shared demo-api:   app.include_router(docqa.api.router, prefix="/a1")
Run it standalone:                 uvicorn docqa.main:app

Abuse protection (env-configurable, see .env.example): gateway-secret header, per-IP rate limit,
daily LLM-call budget, provider allow-list, question length cap. The rate limit and the budget live in
memory: they reset when the process restarts (a free Render instance restarts after every idle spell). Set
API_BUDGET_STORE=db to keep the daily LLM-call budget in the database (table api_usage) so it survives restarts;
the per-IP rate limit is per minute, so keeping it in memory is fine for one instance.
"""
from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Callable

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from .config import ROOT, load_settings
from .generate import AnswerResult, Assistant
from .retrieval import Retriever

log = logging.getLogger("docqa.api")
router = APIRouter()

MAX_QUESTION_CHARS = 500


# ---------------------------------------------------------------------------------------- config
@dataclass(frozen=True)
class ApiConfig:
    gateway_secret: str = ""  # empty = no secret required (local development only)
    default_provider: str = "gemini"
    allowed_providers: tuple[str, ...] = ("gemini", "groq")
    rate_per_minute: int = 10
    daily_llm_calls: int = 300


def load_api_config() -> ApiConfig:
    allowed = tuple(p.strip().lower() for p in os.getenv("API_ALLOWED_PROVIDERS", "gemini,groq").split(",") if p.strip())
    default = os.getenv("API_DEFAULT_PROVIDER", "").strip().lower() or allowed[0]
    return ApiConfig(
        gateway_secret=os.getenv("GATEWAY_SECRET", "").strip(),
        default_provider=default,
        allowed_providers=allowed,
        rate_per_minute=int(os.getenv("API_RATE_LIMIT_PER_MIN", "10")),
        daily_llm_calls=int(os.getenv("API_DAILY_LLM_CALLS", "300")),
    )


# ------------------------------------------------------------------------------------- guards
class RateLimiter:
    """Sliding window: at most `per_minute` requests per key (client IP) in any 60 seconds."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_minute, self.clock = per_minute, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            if len(self._hits) > 5000:  # forget idle clients so memory cannot grow without bound
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= 60]:
                    del self._hits[k]
            return True


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class DailyBudget:
    """Kill-switch: stop calling the LLM after `max_calls` calls per UTC day."""

    def __init__(self, max_calls: int, today: Callable[[], str] = _utc_day) -> None:
        self.max_calls, self.today = max_calls, today
        self.day, self.used = today(), 0
        self._lock = threading.Lock()

    def _roll(self) -> None:
        d = self.today()
        if d != self.day:
            self.day, self.used = d, 0

    def exhausted(self) -> bool:
        with self._lock:
            self._roll()
            return self.used >= self.max_calls

    def add(self, n: int) -> None:
        with self._lock:
            self._roll()
            self.used += n


class PersistentBudget:
    """Daily LLM-call budget stored in the database, so spin-downs and restarts cannot reset it.
    Fails closed: if the database cannot be read, the demo stops spending instead of guessing."""

    def __init__(self, service: Any, max_calls: int) -> None:
        self.service, self.max_calls = service, max_calls

    def exhausted(self) -> bool:
        try:
            return self.service.usage_today() >= self.max_calls
        except Exception:  # noqa: BLE001
            log.exception("could not read the daily budget; refusing to spend")
            return True

    def add(self, n: int) -> None:
        if n <= 0:
            return
        try:
            self.service.add_usage(n)
        except Exception:  # noqa: BLE001
            log.exception("could not record %s LLM call(s) in the daily budget", n)


# ------------------------------------------------------------------------------------ service
def _default_llm_factory(provider: str):
    from llm_client import LLM

    return LLM(provider)


def _build_store_and_retriever(s):
    from .embedder import Embedder
    from .store import Store

    store = Store(s.database_url, s.db_schema)
    emb = Embedder(s.embed_provider, s.embed_model, batch_size=s.embed_batch_size)
    return store, Retriever(store, emb, candidate_k=s.candidate_k, keyword_weight=s.keyword_weight)


_SOURCES_SQL = """SELECT DISTINCT d.doc_id, d.title, d.source_url, d.version, d.effective_date
                  FROM chunks c JOIN documents d USING (doc_id)
                  WHERE c.audience = 'public' AND c.status = 'current'
                  ORDER BY d.title"""


_USAGE_DDL = "CREATE TABLE IF NOT EXISTS api_usage (day date PRIMARY KEY, llm_calls integer NOT NULL DEFAULT 0)"
_USAGE_TODAY = "SELECT llm_calls FROM api_usage WHERE day = (now() AT TIME ZONE 'utc')::date"
_USAGE_ADD = """INSERT INTO api_usage (day, llm_calls) VALUES ((now() AT TIME ZONE 'utc')::date, %s)
                ON CONFLICT (day) DO UPDATE SET llm_calls = api_usage.llm_calls + EXCLUDED.llm_calls
                RETURNING llm_calls"""


class ChatService:
    """Owns the database connection and one Assistant per provider.

    The connection is opened lazily (the API can start before Postgres is ready) and re-opened once if
    it was dropped (Neon closes idle connections). One question is answered at a time: the lock keeps the
    single connection safe; use a connection pool if you ever need real concurrency.
    """

    def __init__(self, settings, build: Callable[[], tuple[Any, Retriever]] | None = None,
                 llm_factory: Callable[[str], Any] | None = None) -> None:
        self.settings = settings
        self._build = build or (lambda: _build_store_and_retriever(settings))
        self._llm_factory = llm_factory or _default_llm_factory
        self.store: Any = None
        self.retriever: Retriever | None = None
        self._assistants: dict[str, Assistant] = {}
        self._lock = threading.Lock()

    def _rebuild(self) -> None:
        if self.store is not None:
            try:
                self.store.close()
            except Exception:  # noqa: BLE001
                pass
        self.store = None
        self._assistants.clear()
        self.store, self.retriever = self._build()

    def _call(self, fn: Callable[[], Any]) -> Any:
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self.store is None or self.store.conn.closed:
                        self._rebuild()
                    return fn()
                except psycopg.OperationalError:
                    if attempt == 2:
                        raise
                    self._rebuild()

    def _assistant(self, provider: str) -> Assistant:
        a = self._assistants.get(provider)
        if a is None:
            s = self.settings
            a = Assistant(self.retriever, self._llm_factory(provider), k=s.answer_k, min_cosine=s.abstain_min_cosine,
                          max_tokens=s.answer_max_tokens, temperature=s.answer_temperature)
            self._assistants[provider] = a
        return a

    # public
    def ping(self) -> bool:
        try:
            return bool(self._call(lambda: self.store.load_meta()))
        except Exception:  # noqa: BLE001
            return False

    def ask(self, provider: str, question: str) -> AnswerResult:
        return self._call(lambda: self._assistant(provider).ask(question))

    def usage_today(self) -> int:
        def run() -> int:
            self.store.conn.execute(_USAGE_DDL)
            row = self.store.conn.execute(_USAGE_TODAY).fetchone()
            return int(row["llm_calls"]) if row else 0

        return self._call(run)

    def add_usage(self, n: int) -> int:
        def run() -> int:
            self.store.conn.execute(_USAGE_DDL)
            return int(self.store.conn.execute(_USAGE_ADD, (n,)).fetchone()["llm_calls"])

        return self._call(run)

    def sources(self) -> list[dict]:
        def run() -> list[dict]:
            rows = self.store.conn.execute(_SOURCES_SQL).fetchall()
            return [{"doc_id": r["doc_id"], "title": r["title"], "source_url": r["source_url"] or "",
                     "version": str(r["version"] or ""), "effective_date": str(r["effective_date"] or "")}
                    for r in rows]

        return self._call(run)


@dataclass
class Runtime:
    service: Any  # ChatService, or a test double with ping() / ask() / sources()
    config: ApiConfig
    limiter: RateLimiter
    budget: Any  # DailyBudget (memory) or PersistentBudget (database)


@lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    s = load_settings()  # also loads .env, so it must run before load_api_config()
    if not os.environ.get("LLM_PRICES_FILE"):
        os.environ["LLM_PRICES_FILE"] = str(ROOT / "prices.json")
    cfg = load_api_config()
    service = ChatService(s)
    if os.getenv("API_BUDGET_STORE", "memory").strip().lower() == "db":
        budget: Any = PersistentBudget(service, cfg.daily_llm_calls)
    else:
        budget = DailyBudget(cfg.daily_llm_calls)
    return Runtime(service, cfg, RateLimiter(cfg.rate_per_minute), budget)


def require_gateway(request: Request, rt: Runtime = Depends(get_runtime)) -> Runtime:
    secret = rt.config.gateway_secret
    if secret:
        given = request.headers.get("x-gateway-secret", "")
        if not hmac.compare_digest(given.encode(), secret.encode()):
            raise HTTPException(status_code=401, detail="Invalid or missing gateway secret.")
    return rt


def _client_ip(request: Request) -> str:
    # The edge function sets X-Real-Client-IP; it is only trusted because the secret was checked first.
    forwarded = request.headers.get("x-real-client-ip", "").strip()
    return forwarded or (request.client.host if request.client else "unknown")


# --------------------------------------------------------------------------------------- schema
class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)
    provider: str | None = None

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question is too short")
        return v


class CitationOut(BaseModel):
    id: str
    doc_id: str
    title: str
    heading: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    abstained: bool
    reason: str
    citations: list[CitationOut]
    provider: str | None = None
    model: str | None = None
    latency_s: float = 0.0
    cost_usd: float = 0.0


class SourceOut(BaseModel):
    doc_id: str
    title: str
    source_url: str
    version: str
    effective_date: str


# ------------------------------------------------------------------------------------ endpoints
@router.get("/health")
def health(rt: Runtime = Depends(get_runtime)) -> dict:
    if not rt.service.ping():
        raise HTTPException(status_code=503, detail="Not ready: database unreachable or empty (run: docqa ingest).")
    return {"status": "ok"}


@router.get("/sources", response_model=list[SourceOut])
def sources(rt: Runtime = Depends(require_gateway)) -> list[dict]:
    return rt.service.sources()


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request, rt: Runtime = Depends(require_gateway)) -> ChatResponse:
    if not rt.limiter.allow(_client_ip(request)):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute.")
    if rt.budget.exhausted():
        raise HTTPException(status_code=503, detail="The demo's daily AI budget is used up. Try again tomorrow.")
    provider = (body.provider or rt.config.default_provider).strip().lower()
    if provider not in rt.config.allowed_providers:
        raise HTTPException(status_code=400, detail=f"provider must be one of: {', '.join(rt.config.allowed_providers)}")

    try:
        res: AnswerResult = rt.service.ask(provider, body.question)
    except Exception:  # noqa: BLE001  never leak internals to the visitor
        log.exception("chat failed (provider=%s)", provider)
        raise HTTPException(status_code=503, detail="The assistant is temporarily unavailable.") from None

    rt.budget.add(res.llm_calls)
    log.info("chat provider=%s reason=%s llm_calls=%s latency=%.2fs", provider, res.reason, res.llm_calls, res.latency_s)
    if res.reason in ("llm_error", "invalid_json"):
        raise HTTPException(status_code=502, detail="The AI provider failed. Try again or switch provider.")
    return ChatResponse(
        answer=res.answer, abstained=res.abstained, reason=res.reason,
        citations=[CitationOut(id=c.id, doc_id=c.doc_id, title=c.title, heading=c.heading, url=c.url)
                   for c in res.citations],
        provider=res.provider, model=res.model, latency_s=round(res.latency_s, 2), cost_usd=round(res.cost_usd, 6),
    )
