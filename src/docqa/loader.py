"""Load markdown documents with YAML front matter and validate their metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

REQUIRED = ("title", "doc_type", "version", "status", "audience")
STATUSES = {"current", "deprecated"}
AUDIENCES = {"public", "internal"}

_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.S)


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    body: str
    meta: dict


def parse_document(doc_id: str, raw: str) -> Document:
    m = _FRONT.match(raw)
    if not m:
        raise ValueError(f"{doc_id}: missing YAML front matter")
    meta = yaml.safe_load(m.group(1)) or {}
    missing = [k for k in REQUIRED if k not in meta]
    if missing:
        raise ValueError(f"{doc_id}: missing front matter fields {missing}")
    if meta["status"] not in STATUSES:
        raise ValueError(f"{doc_id}: status must be one of {sorted(STATUSES)}")
    if meta["audience"] not in AUDIENCES:
        raise ValueError(f"{doc_id}: audience must be one of {sorted(AUDIENCES)}")
    meta = {k: (str(v) if k in ("version", "effective_date") else v) for k, v in meta.items()}
    return Document(doc_id=doc_id, title=str(meta["title"]), body=m.group(2).strip(), meta=meta)


def load_corpus(directory: Path) -> list[Document]:
    docs = [parse_document(p.stem, p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.md"))]
    if not docs:
        raise FileNotFoundError(f"no .md documents found in {directory}")
    return docs
