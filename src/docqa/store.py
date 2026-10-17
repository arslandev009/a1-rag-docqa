"""Postgres + pgvector store: documents, chunks (vector + full-text), and filtered search.

* Embeddings are stored as `halfvec` when the model dimension is above 2000 (pgvector's index
  limit for `vector`), otherwise `vector`. Gemini's 3072-dim embeddings therefore use halfvec.
* `status` and `audience` are copied onto every chunk so retrieval can filter deprecated and
  internal content in SQL, before ranking.
* A `meta` table remembers the embedding model and dimension so a model change is detected
  instead of silently mixing vector spaces.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import psycopg
from pgvector import HalfVector, Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from .chunker import Chunk
from .loader import Document

_SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_WORD = re.compile(r"[A-Za-z0-9_]+")
HALFVEC_THRESHOLD = 2000


class SchemaMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    heading_path: str
    content: str
    source_url: str
    status: str
    audience: str
    score: float
    version: str = ""
    effective_date: str = ""


_HIT_COLS = ("c.chunk_id, c.doc_id, d.title, c.heading_path, c.content, d.source_url, c.status, c.audience, "
             "d.version, d.effective_date")


def _or_tsquery(text: str) -> str:
    """Natural-language question -> OR-ed tsquery (plain AND matching returns nothing for questions)."""
    return " | ".join(dict.fromkeys(w.lower() for w in _WORD.findall(text)))


def _hit(row: dict) -> Hit:
    return Hit(row["chunk_id"], row["doc_id"], row["title"], row["heading_path"], row["content"],
               row["source_url"], row["status"], row["audience"], float(row["score"]),
               row.get("version") or "", row.get("effective_date") or "")


class Store:
    def __init__(self, database_url: str, schema: str = "docqa") -> None:
        if not _SCHEMA_RE.match(schema):
            raise ValueError(f"invalid schema name: {schema!r}")
        self.schema = schema
        self.dim: int | None = None
        self.conn = psycopg.connect(database_url, autocommit=True, row_factory=dict_row)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        self.conn.execute(f"SET search_path TO {schema}, public")
        register_vector(self.conn)
        try:  # pgvector >= 0.8: keep filtered HNSW searches from under-returning
            self.conn.execute("SET hnsw.iterative_scan = strict_order")
        except psycopg.Error:
            pass

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self.conn.close()

    def transaction(self):
        return self.conn.transaction()

    def drop_schema(self) -> None:
        self.conn.execute(f"DROP SCHEMA IF EXISTS {self.schema} CASCADE")

    def load_meta(self) -> dict[str, str]:
        exists = self.conn.execute("SELECT to_regclass('meta') IS NOT NULL AS ok").fetchone()["ok"]
        if not exists:
            return {}
        meta = {r["key"]: r["value"] for r in self.conn.execute("SELECT key, value FROM meta")}
        if "embed_dim" in meta:
            self.dim = int(meta["embed_dim"])
        return meta

    def init_schema(self, dim: int, model: str, reset: bool = False) -> None:
        if reset:
            self.conn.execute("DROP TABLE IF EXISTS chunks, documents, meta CASCADE")
        self.conn.execute("CREATE TABLE IF NOT EXISTS meta (key text PRIMARY KEY, value text NOT NULL)")
        stored = self.load_meta()
        if stored and (stored.get("embed_dim") != str(dim) or stored.get("embed_model") != model):
            raise SchemaMismatch(
                f"Database was built with model={stored.get('embed_model')} dim={stored.get('embed_dim')}, "
                f"but you are using model={model} dim={dim}. Re-run ingest with --reset."
            )
        coltype = "halfvec" if dim > HALFVEC_THRESHOLD else "vector"
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS documents (
                 doc_id text PRIMARY KEY, title text NOT NULL, doc_type text, version text,
                 status text NOT NULL, audience text NOT NULL, effective_date text,
                 source_url text, superseded_by text, updated_at timestamptz NOT NULL DEFAULT now())"""
        )
        self.conn.execute(
            f"""CREATE TABLE IF NOT EXISTS chunks (
                  chunk_id text PRIMARY KEY,
                  doc_id text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                  chunk_index int NOT NULL, heading_path text NOT NULL, content text NOT NULL,
                  content_hash text NOT NULL, status text NOT NULL, audience text NOT NULL,
                  embedding {coltype}({dim}) NOT NULL,
                  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED)"""
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS chunks_filter_idx ON chunks (audience, status)")
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding {coltype}_cosine_ops)"
        )
        for k, v in (("embed_model", model), ("embed_dim", str(dim))):
            self.conn.execute(
                "INSERT INTO meta (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (k, v),
            )
        self.dim = dim

    # ------------------------------------------------------------------ writes
    def _vec(self, values: Sequence[float]):
        assert self.dim is not None, "call init_schema() or load_meta() first"
        return HalfVector(values) if self.dim > HALFVEC_THRESHOLD else Vector(values)

    def upsert_document(self, doc: Document) -> None:
        m = doc.meta
        self.conn.execute(
            """INSERT INTO documents (doc_id, title, doc_type, version, status, audience, effective_date, source_url, superseded_by)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (doc_id) DO UPDATE SET title=EXCLUDED.title, doc_type=EXCLUDED.doc_type,
                 version=EXCLUDED.version, status=EXCLUDED.status, audience=EXCLUDED.audience,
                 effective_date=EXCLUDED.effective_date, source_url=EXCLUDED.source_url,
                 superseded_by=EXCLUDED.superseded_by, updated_at=now()""",
            (doc.doc_id, doc.title, m.get("doc_type"), m.get("version"), m["status"], m["audience"],
             m.get("effective_date"), m.get("source_url"), m.get("superseded_by")),
        )

    def chunk_hashes(self, doc_id: str) -> dict[str, str]:
        rows = self.conn.execute("SELECT chunk_id, content_hash FROM chunks WHERE doc_id = %s", (doc_id,))
        return {r["chunk_id"]: r["content_hash"] for r in rows}

    def upsert_chunk(self, c: Chunk, embedding: Sequence[float], content_hash: str) -> None:
        self.conn.execute(
            """INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, content, content_hash, status, audience, embedding)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (chunk_id) DO UPDATE SET chunk_index=EXCLUDED.chunk_index, heading_path=EXCLUDED.heading_path,
                 content=EXCLUDED.content, content_hash=EXCLUDED.content_hash, status=EXCLUDED.status,
                 audience=EXCLUDED.audience, embedding=EXCLUDED.embedding""",
            (c.chunk_id, c.doc_id, c.index, c.heading_path, c.text, content_hash,
             c.meta["status"], c.meta["audience"], self._vec(embedding)),
        )

    def update_chunk_meta(self, c: Chunk) -> None:
        """Text unchanged: refresh metadata (status/audience/position) without re-embedding."""
        self.conn.execute(
            "UPDATE chunks SET chunk_index=%s, heading_path=%s, status=%s, audience=%s WHERE chunk_id=%s",
            (c.index, c.heading_path, c.meta["status"], c.meta["audience"], c.chunk_id),
        )

    def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        if not chunk_ids:
            return 0
        return self.conn.execute("DELETE FROM chunks WHERE chunk_id = ANY(%s)", (list(chunk_ids),)).rowcount

    def delete_docs_not_in(self, keep: Sequence[str]) -> int:
        return self.conn.execute("DELETE FROM documents WHERE NOT (doc_id = ANY(%s))", (list(keep),)).rowcount

    # ------------------------------------------------------------------ search
    def vector_search(self, query_vec: Sequence[float], k: int = 5,
                      audiences: Sequence[str] = ("public",), statuses: Sequence[str] = ("current",)) -> list[Hit]:
        v = self._vec(query_vec)
        rows = self.conn.execute(
            f"""SELECT {_HIT_COLS}, 1 - (c.embedding <=> %s) AS score
                FROM chunks c JOIN documents d USING (doc_id)
                WHERE c.audience = ANY(%s) AND c.status = ANY(%s)
                ORDER BY c.embedding <=> %s LIMIT %s""",
            (v, list(audiences), list(statuses), v, k),
        ).fetchall()
        return [_hit(r) for r in rows]

    def keyword_search(self, query: str, k: int = 5,
                       audiences: Sequence[str] = ("public",), statuses: Sequence[str] = ("current",)) -> list[Hit]:
        tsq = _or_tsquery(query)
        if not tsq:
            return []
        rows = self.conn.execute(
            f"""SELECT {_HIT_COLS}, ts_rank_cd(c.tsv, q) AS score
                FROM chunks c JOIN documents d USING (doc_id), to_tsquery('english', %s) q
                WHERE c.tsv @@ q AND c.audience = ANY(%s) AND c.status = ANY(%s)
                ORDER BY score DESC LIMIT %s""",
            (tsq, list(audiences), list(statuses), k),
        ).fetchall()
        return [_hit(r) for r in rows]

    # ------------------------------------------------------------------ info
    def stats(self) -> dict[str, Any]:
        meta = self.load_meta()
        if not meta:
            return {"meta": {}, "documents": 0, "chunks": 0, "by_status": {}}
        by = self.conn.execute(
            "SELECT audience, status, count(*) AS n FROM chunks GROUP BY audience, status ORDER BY 1, 2"
        ).fetchall()
        return {
            "meta": meta,
            "documents": self.conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"],
            "chunks": self.conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"],
            "by_status": {f"{r['audience']}/{r['status']}": r["n"] for r in by},
        }
