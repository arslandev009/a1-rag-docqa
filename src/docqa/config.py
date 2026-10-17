from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path
    eval_path: Path
    database_url: str
    db_schema: str
    embed_provider: str
    embed_model: str
    embed_batch_size: int
    chunk_max_chars: int
    chunk_overlap_sentences: int
    candidate_k: int
    keyword_weight: float
    answer_k: int
    abstain_min_cosine: float
    answer_max_tokens: int
    answer_temperature: float | None


def _temperature() -> float | None:
    raw = os.getenv("ANSWER_TEMPERATURE", "0").strip()
    return float(raw) if raw else None  # empty string = do not send the parameter


def load_settings() -> Settings:
    load_dotenv(find_dotenv(usecwd=True))
    return Settings(
        corpus_dir=ROOT / "data" / "corpus",
        eval_path=ROOT / "data" / "eval" / "questions.jsonl",
        database_url=os.getenv("DATABASE_URL", "postgresql://docqa:docqa_dev_password@127.0.0.1:55433/docqa"),
        db_schema=os.getenv("DB_SCHEMA", "docqa"),
        embed_provider=os.getenv("EMBED_PROVIDER", "gemini"),
        embed_model=os.getenv("EMBED_MODEL", ""),
        embed_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "32")),
        chunk_max_chars=int(os.getenv("CHUNK_MAX_CHARS", "1200")),
        chunk_overlap_sentences=int(os.getenv("CHUNK_OVERLAP_SENTENCES", "1")),
        candidate_k=int(os.getenv("RETRIEVAL_CANDIDATE_K", "20")),
        keyword_weight=float(os.getenv("RETRIEVAL_KEYWORD_WEIGHT", "0.25")),
        answer_k=int(os.getenv("ANSWER_K", "5")),
        abstain_min_cosine=float(os.getenv("ABSTAIN_MIN_COSINE", "0.55")),
        answer_max_tokens=int(os.getenv("ANSWER_MAX_TOKENS", "1500")),
        answer_temperature=_temperature(),
    )
