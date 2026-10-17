"""Heading-aware chunking.

Each chunk keeps its heading path ("Title > Section") as a header line, so the
embedding and keyword index both see the context a human would. Metadata
(status, audience, version, source_url) travels with every chunk so retrieval can
filter deprecated or internal content *before* ranking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .loader import Document

_HEADING = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    heading_path: str
    text: str  # header + body; this is what gets embedded and indexed
    meta: dict


def split_sections(body: str, title: str) -> list[tuple[str, str]]:
    """Return [(heading_path, text)] for each non-empty section."""
    sections: list[tuple[str, str]] = []
    path: list[tuple[int, str]] = []
    buf: list[str] = []
    current = title

    for line in body.splitlines():
        m = _HEADING.match(line)
        if m:
            text = "\n".join(buf).strip()
            if text:
                sections.append((current, text))
            buf = []
            level, name = len(m.group(1)), m.group(2)
            path = [p for p in path if p[0] < level] + [(level, name)]
            names = [n for lvl, n in path if lvl > 1]  # the H1 duplicates the front-matter title
            current = " > ".join([title, *names]) if names else title
        else:
            buf.append(line)
    text = "\n".join(buf).strip()
    if text:
        sections.append((current, text))
    return sections


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE.split(text.strip()) if s]


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    cur = ""
    for s in _sentences(paragraph):
        while len(s) > max_chars:  # pathological sentence: hard slice
            pieces.append(s[:max_chars])
            s = s[max_chars:]
        if cur and len(cur) + 1 + len(s) > max_chars:
            pieces.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def pack(text: str, max_chars: int, overlap_sentences: int = 1) -> list[str]:
    """Pack paragraphs into chunks of at most ~max_chars, repeating the last
    sentence(s) of the previous chunk so ideas split across chunks stay findable."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paragraphs:
        for piece in ([p] if len(p) <= max_chars else _split_long(p, max_chars)):
            if cur and len(cur) + 2 + len(piece) > max_chars:
                chunks.append(cur)
                tail = " ".join(_sentences(cur)[-overlap_sentences:]) if overlap_sentences else ""
                cur = f"{tail}\n\n{piece}" if tail else piece
            else:
                cur = f"{cur}\n\n{piece}" if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


def chunk_document(doc: Document, max_chars: int = 1200, overlap_sentences: int = 1) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading_path, text in split_sections(doc.body, doc.title):
        for body in pack(text, max_chars, overlap_sentences):
            idx = len(chunks)
            meta = {**doc.meta, "doc_id": doc.doc_id, "heading_path": heading_path}
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{idx}",
                    doc_id=doc.doc_id,
                    index=idx,
                    heading_path=heading_path,
                    text=f"{heading_path}\n\n{body}",
                    meta=meta,
                )
            )
    return chunks
