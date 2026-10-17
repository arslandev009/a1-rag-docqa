"""Chunk-level retrieval evaluation (A1.6c). No LLM, no API cost (query vectors come from the cache).

Why: eval_retrieval.py counts a hit when *any chunk of the gold document* is retrieved. For q44 that hid the real
failure: the Billing chunk of pricing-plans was found, the Plans chunk (the one with "$19/month") was not.
Here a question only counts as covered when the *gold chunks* are in the context the model would see.

Gold chunks: data/eval/questions.jsonl labels documents only, so gold chunks are derived from each question's
`expected` text (numbers and rarer words that appear in a chunk; every gold document contributes its best chunk).
The result is written to data/eval/gold_chunks.json: review it, edit `gold_chunks` by hand where it is wrong and set
`reviewed` to true. Until then every number here is "auto-labelled, unreviewed".

Metrics per configuration (hybrid retrieval, answer_k anchors, optional neighbour expansion):
  anchor_any  share of questions with at least one gold chunk among the top-k hits
  ctx_any     ... in the final context (hits + neighbours)
  ctx_all     share of questions whose gold chunks are ALL in the final context (the model can answer fully)
  avg_chunks  mean number of chunks sent to the model (cost driver)
  violations  deprecated or internal chunks in any context (must be 0)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from .retrieval import Retriever

_NUM = re.compile(r"\$?\d[\w,.\-%]*")
_WORD = re.compile(r"[a-z][a-z_]{3,}")
STOP = frozenset(
    "that this with from have does your they them what when which where there their would could should about into than "
    "then also only each such will been being more most some other these those after before while using used uses "
    "cost costs".split())


def key_tokens(expected: str) -> dict[str, float]:
    """Numbers/prices weigh 2, words 1 (stop words dropped)."""
    text = expected.lower()
    tokens: dict[str, float] = {}
    for m in _NUM.findall(text):
        t = m.strip(".,;:")
        if t:
            tokens[t] = 2.0
    for w in _WORD.findall(text):
        if w not in STOP:
            tokens.setdefault(w, 1.0)
    return tokens


def _pattern(token: str) -> re.Pattern:
    if token[0].isdigit() or token[0] == "$":  # numbers match whole ("50" must not match "500")
        return re.compile(r"(?<![\w.$])" + re.escape(token) + r"(?!\w)")
    return re.compile(r"\b" + re.escape(token))  # words match as a prefix ("refund" matches "refundable")


def auto_label(expected: str, chunks: Sequence[dict]) -> list[str]:
    """chunks: dicts with chunk_id, doc_id, chunk_index, content (the gold documents' public, current chunks)."""
    toks = key_tokens(expected)
    present: dict[str, set[str]] = {}
    for t in toks:
        pat = _pattern(t)
        present[t] = {c["chunk_id"] for c in chunks if pat.search(c["content"].lower())}
    scores = {c["chunk_id"]: sum(w / len(present[t]) for t, w in toks.items() if c["chunk_id"] in present[t])
              for c in chunks}
    best = max(scores.values(), default=0.0)
    if best <= 0:
        return []
    chosen = {cid for cid, s in scores.items() if s >= 0.5 * best}
    for doc in sorted({c["doc_id"] for c in chunks}):  # every gold document contributes its best chunk
        mine = [c for c in chunks if c["doc_id"] == doc]
        top = max(mine, key=lambda c: (scores[c["chunk_id"]], -c["chunk_index"]))
        if scores[top["chunk_id"]] > 0:
            chosen.add(top["chunk_id"])
    order = {c["chunk_id"]: (c["doc_id"], c["chunk_index"]) for c in chunks}
    return sorted(chosen, key=lambda cid: order[cid])


def fetch_public_chunks(store, doc_ids: Sequence[str]) -> list[dict]:
    return store.conn.execute(
        """SELECT chunk_id, doc_id, chunk_index, heading_path, content FROM chunks
           WHERE doc_id = ANY(%s) AND audience = 'public' AND status = 'current'
           ORDER BY doc_id, chunk_index""", (list(doc_ids),)).fetchall()


def labels_for(path: Path, store, questions: Sequence[dict], relabel: bool = False) -> dict[str, dict]:
    if path.exists() and not relabel:
        return json.loads(path.read_text(encoding="utf-8"))["labels"]
    docs = sorted({d for q in questions for d in q.get("sources", [])})
    chunks = fetch_public_chunks(store, docs)
    labels: dict[str, dict] = {}
    for q in questions:
        if not q.get("answerable") or not q.get("sources"):
            continue
        cand = [c for c in chunks if c["doc_id"] in q["sources"]]
        labels[q["id"]] = {"question": q["question"], "expected": q.get("expected", ""),
                           "gold_chunks": auto_label(q.get("expected", ""), cand), "reviewed": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"note": "auto-labelled by eval_chunks.auto_label; fix gold_chunks by hand and set reviewed=true",
                                "labels": labels}, indent=2), encoding="utf-8")
    return labels


def grid(store, embedder, questions: Sequence[dict], labels: dict[str, dict], vecs: dict[str, list[float]], *,
         weights: Sequence[float] = (0.0, 0.1, 0.25, 0.5), candidate_ks: Sequence[int] = (10, 20),
         ks: Sequence[int] = (5, 8), radii: Sequence[int] = (0, 1), neighbor_top: int = 3) -> list[dict]:
    qs = [q for q in questions if labels.get(q["id"], {}).get("gold_chunks")]
    kmax = max(ks)
    rows: list[dict] = []
    for ck in candidate_ks:
        for w in weights:
            r = Retriever(store, embedder, candidate_k=ck, keyword_weight=w, neighbors=0, neighbor_top=neighbor_top)
            ranked = {q["id"]: r.retrieve(q["question"], k=kmax, mode="hybrid", query_vec=vecs.get(q["question"])).items
                      for q in qs}
            for k in ks:
                for rad in radii:
                    r.neighbors = rad
                    any_a = any_c = all_c = viol = size = 0
                    missing: list[str] = []
                    for q in qs:
                        gold = set(labels[q["id"]]["gold_chunks"])
                        anchors = ranked[q["id"]][:k]
                        ctx = r.expand(anchors)
                        a_ids, c_ids = {x.chunk_id for x in anchors}, {x.chunk_id for x in ctx}
                        any_a += bool(gold & a_ids)
                        any_c += bool(gold & c_ids)
                        if gold <= c_ids:
                            all_c += 1
                        else:
                            missing.append(q["id"])
                        viol += sum(1 for x in ctx if x.hit.audience != "public" or x.hit.status != "current")
                        size += len(ctx)
                    n = len(qs) or 1
                    rows.append({"candidate_k": ck, "keyword_weight": w, "k": k, "neighbors": rad,
                                 "anchor_any": any_a / n, "ctx_any": any_c / n, "ctx_all": all_c / n,
                                 "avg_chunks": size / n, "violations": viol, "missing": missing, "n": len(qs)})
    rows.sort(key=lambda x: (-x["ctx_all"], x["avg_chunks"], x["candidate_k"], x["keyword_weight"]))
    return rows


def _row(r: dict) -> str:
    return (f"| {r['candidate_k']} | {r['keyword_weight']} | {r['k']} | {r['neighbors']} | {r['anchor_any']:.3f} | "
            f"{r['ctx_any']:.3f} | {r['ctx_all']:.3f} | {r['avg_chunks']:.1f} | {r['violations']} |")


_HEAD = ["| candidate_k | keyword weight | k | neighbours | anchor_any | ctx_any | ctx_all | avg chunks | violations |",
         "|---|---|---|---|---|---|---|---|---|"]


# configurations always shown (candidate_k, keyword weight, k, neighbours); the first one is the current default
WATCH = [(20, 0.25, 5, 0), (20, 0.25, 5, 1), (20, 0.1, 5, 0), (20, 0.1, 5, 1), (20, 0.0, 5, 1), (20, 0.1, 8, 1)]


def run(store, embedder, questions: Sequence[dict], vecs: dict[str, list[float]], labels_path: Path, *,
        relabel: bool = False) -> str:
    labels = labels_for(labels_path, store, questions, relabel)
    docs = sorted({d for q in questions for d in q.get("sources", [])})
    heads = {c["chunk_id"]: c["heading_path"].split(" > ")[-1] for c in fetch_public_chunks(store, docs)}
    rows = grid(store, embedder, questions, labels, vecs)
    n = rows[0]["n"] if rows else 0
    unlabeled = [qid for qid, v in labels.items() if not v["gold_chunks"]]

    lines = ["# Chunk-level retrieval evaluation", "",
             f"{n} answerable questions with gold chunks ({len(labels)} answerable in total). "
             "**Gold chunks are auto-labelled from each question's `expected` text unless marked reviewed.**", "",
             "`ctx_all` = share of questions whose gold chunks are all in the context the model sees; "
             "`avg chunks` = chunks sent to the model (cost).", ""]
    if unlabeled:
        lines += [f"No gold chunk could be derived for: {', '.join(unlabeled)} (excluded).", ""]

    by = {(r["candidate_k"], r["keyword_weight"], r["k"], r["neighbors"]): r for r in rows}
    watch = [by[w] for w in WATCH if w in by]
    reviewed = sum(1 for v in labels.values() if v.get("reviewed"))
    lines += [f"Labels reviewed by a human: {reviewed} of {len(labels)}.", "",
              "## Candidate configurations (first row = current default)", ""] + _HEAD
    lines += [_row(r) for r in watch]
    lines += ["", "## Best 12 configurations (sorted by ctx_all, then fewest chunks)", ""] + _HEAD
    lines += [_row(r) for r in rows[:12]]
    lines += ["", "## Questions whose gold chunks are NOT all in the context (candidates)", ""]
    lines += [f"- ck={r['candidate_k']} w={r['keyword_weight']} k={r['k']} neighbours={r['neighbors']}: "
              f"{', '.join(r['missing']) or 'none'}" for r in watch]
    lines += ["", "## Gold chunk labels (review these; edit data/eval/gold_chunks.json)", "",
              "| id | expected | gold chunks |", "|---|---|---|"]
    for qid, v in labels.items():
        golds = ", ".join(f"{c} [{heads.get(c, '?')}]" for c in v["gold_chunks"]) or "(none)"
        lines.append(f"| {qid} | {v['expected'][:70]} | {golds} |")
    return "\n".join(lines) + "\n"
