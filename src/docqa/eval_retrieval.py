"""Retrieval evaluation over the labelled question set (no LLM involved).

For every answerable question with gold source documents we measure, per retrieval mode:
  hit@k     did any gold document appear in the top-k chunks?
  recall@k  what share of the gold documents appeared (matters for multi-document questions)
  MRR       1 / rank of the first gold chunk (how early the right evidence shows up)
and we count policy violations (a deprecated or internal chunk in any result: must be 0).
For vector search it also records the best cosine similarity of answerable vs unanswerable questions,
which is used in A1.4 to choose the "I couldn't find this" threshold.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .retrieval import MODES, Retriever


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rank_metrics(ranked_docs: Sequence[str], gold: set[str], ks: Sequence[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        top = set(ranked_docs[:k])
        out[f"hit@{k}"] = 1.0 if gold & top else 0.0
        out[f"recall@{k}"] = len(gold & top) / len(gold)
    out["mrr"] = next((1.0 / i for i, d in enumerate(ranked_docs, 1) if d in gold), 0.0)
    return out


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def embed_queries(embedder, questions: Sequence[dict], cache_path: Path) -> dict[str, list[float]]:
    """Embed every question once; vectors are cached on disk keyed by model + text."""
    cache: dict[str, list[float]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    key = lambda text: hashlib.sha256(f"{embedder.model}\n{text}".encode()).hexdigest()  # noqa: E731
    missing = [q["question"] for q in questions if key(q["question"]) not in cache]
    if missing:
        for text, vec in zip(missing, embedder.embed(missing)):
            cache[key(text)] = vec
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return {q["question"]: cache[key(q["question"])] for q in questions}


def evaluate(retriever: Retriever, questions: Sequence[dict], vecs: dict[str, list[float]] | None = None,
             modes: Sequence[str] = MODES, ks: Sequence[int] = (1, 3, 5)) -> dict:
    kmax = max(ks)
    vecs = vecs or {}
    report: dict = {"ks": list(ks), "questions": len(questions), "modes": {}}
    for mode in modes:
        scored: list[dict[str, float]] = []
        by_type: dict[str, list[dict[str, float]]] = defaultdict(list)
        violations, failures = 0, []
        top_vec: dict[str, list[float]] = {"answerable": [], "unanswerable": []}
        for q in questions:
            res = retriever.retrieve(q["question"], k=kmax, mode=mode, query_vec=vecs.get(q["question"]))
            violations += sum(1 for r in res.items if r.hit.audience != "public" or r.hit.status != "current")
            if mode in ("vector", "hybrid") and res.top_vector_score is not None:
                top_vec["answerable" if q["answerable"] and q["sources"] else "unanswerable"].append(res.top_vector_score)
            if not q["sources"]:
                continue
            m = rank_metrics([r.doc_id for r in res.items], set(q["sources"]), ks)
            scored.append(m)
            by_type[q["type"]].append(m)
            if m[f"hit@{kmax}"] == 0:
                failures.append({"id": q["id"], "question": q["question"], "gold": q["sources"],
                                 "got": [r.chunk_id for r in res.items]})
        entry: dict = {
            "scored_questions": len(scored),
            "metrics": {k: round(_mean(scored, k), 3) for k in scored[0]} if scored else {},
            "by_type": {t: {"n": len(rows), f"hit@{kmax}": round(_mean(rows, f"hit@{kmax}"), 3), "mrr": round(_mean(rows, "mrr"), 3)}
                        for t, rows in sorted(by_type.items())},
            "policy_violations": violations,
            "failures": failures,
        }
        if top_vec["answerable"] and top_vec["unanswerable"]:
            a, u = top_vec["answerable"], top_vec["unanswerable"]
            entry["vector_top1_cosine"] = {
                "answerable": {"min": round(min(a), 3), "median": round(statistics.median(a), 3), "max": round(max(a), 3)},
                "unanswerable": {"min": round(min(u), 3), "median": round(statistics.median(u), 3), "max": round(max(u), 3)},
            }
        report["modes"][mode] = entry
    return report


def to_markdown(report: dict, title: str = "Retrieval evaluation") -> str:
    ks = report["ks"]
    kmax = max(ks)
    lines = [f"# {title}", "",
             f"{report['questions']} labelled questions; metrics are averaged over those with gold source documents.", "",
             "| Mode | " + " | ".join(f"hit@{k}" for k in ks) + " | recall@" + str(kmax) + " | MRR | policy violations |",
             "|---|" + "---|" * (len(ks) + 3)]
    for mode, e in report["modes"].items():
        m = e["metrics"]
        lines.append(f"| {mode} | " + " | ".join(f"{m.get(f'hit@{k}', 0):.3f}" for k in ks)
                     + f" | {m.get(f'recall@{kmax}', 0):.3f} | {m.get('mrr', 0):.3f} | {e['policy_violations']} |")
    for mode, e in report["modes"].items():
        lines += ["", f"## By question type: {mode}", "", f"| Type | n | hit@{kmax} | MRR |", "|---|---|---|---|"]
        lines += [f"| {t} | {v['n']} | {v[f'hit@{kmax}']:.3f} | {v['mrr']:.3f} |" for t, v in e["by_type"].items()]
        if e["failures"]:
            lines += ["", f"Misses (gold document not in top {kmax}):", ""]
            lines += [f"- {f['id']}: {f['question']} (gold: {', '.join(f['gold'])})" for f in e["failures"]]
        if "vector_top1_cosine" in e:
            c = e["vector_top1_cosine"]
            lines += ["", "Top-1 vector cosine, used to choose the abstention threshold in A1.4:", "",
                      "| Questions | min | median | max |", "|---|---|---|---|"]
            lines += [f"| {name} | {v['min']} | {v['median']} | {v['max']} |" for name, v in c.items()]
    return "\n".join(lines) + "\n"


def sweep(store, embedder, questions: Sequence[dict], vecs: dict[str, list[float]], *,
          weights: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0), candidate_ks: Sequence[int] = (5, 10, 20),
          rrf_k: int = 60) -> list[dict]:
    """Grid-search the hybrid knobs. Uses cached query vectors, so it costs no API calls."""
    rows = []
    for ck in candidate_ks:
        for w in weights:
            retriever = Retriever(store, embedder, candidate_k=ck, rrf_k=rrf_k, keyword_weight=w)
            m = evaluate(retriever, questions, vecs, modes=("hybrid",), ks=(1, 3, 5))["modes"]["hybrid"]["metrics"]
            rows.append({"candidate_k": ck, "keyword_weight": w, "hit@1": m["hit@1"], "hit@5": m["hit@5"],
                         "recall@5": m["recall@5"], "mrr": m["mrr"]})
    rows.sort(key=lambda r: (-r["mrr"], -r["recall@5"], r["candidate_k"], r["keyword_weight"]))
    return rows


def sweep_markdown(rows: list[dict], n_questions: int) -> str:
    lines = ["# Hybrid retrieval tuning", "",
             f"Grid over keyword weight and candidate_k on {n_questions} labelled questions (sorted by MRR, then recall@5).",
             "**Caution:** the question set is small; prefer a simple setting over the single best row, and re-check "
             "whenever the corpus grows.", "",
             "| candidate_k | keyword weight | hit@1 | hit@5 | recall@5 | MRR |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['candidate_k']} | {r['keyword_weight']} | {r['hit@1']:.3f} | {r['hit@5']:.3f} | {r['recall@5']:.3f} | {r['mrr']:.3f} |"
              for r in rows]
    return "\n".join(lines) + "\n"


def calibrate(retriever: Retriever, questions: Sequence[dict], vecs: dict[str, list[float]],
              thresholds: Sequence[float] = tuple(round(0.50 + 0.025 * i, 3) for i in range(11))) -> dict:
    """How well could a plain similarity gate separate answerable from unanswerable questions?
    wrongly_blocked = answerable questions whose best cosine is below t (would be refused without asking the model)
    caught          = unanswerable questions whose best cosine is below t (correctly refused for free)"""
    ans, unans = [], []
    for q in questions:
        res = retriever.retrieve(q["question"], k=5, mode="hybrid", query_vec=vecs.get(q["question"]))
        if res.top_vector_score is None:
            continue
        row = {"id": q["id"], "type": q["type"], "question": q["question"], "cosine": round(res.top_vector_score, 3)}
        if q["answerable"] and q["sources"]:
            ans.append(row)
        elif q["type"] == "unanswerable":
            unans.append(row)
    table = [{"threshold": t, "wrongly_blocked": sum(1 for r in ans if r["cosine"] < t),
              "caught": sum(1 for r in unans if r["cosine"] < t)} for t in thresholds]
    return {"answerable": sorted(ans, key=lambda r: r["cosine"]), "unanswerable": sorted(unans, key=lambda r: r["cosine"]),
            "table": table}


def calibration_markdown(cal: dict) -> str:
    na, nu = len(cal["answerable"]), len(cal["unanswerable"])
    lines = ["# Abstention gate calibration", "",
             f"Best vector cosine per question ({na} answerable, {nu} unanswerable). A similarity gate refuses a question "
             "without calling the model when the best cosine is below the threshold.", "",
             f"| threshold | answerable wrongly blocked (of {na}) | unanswerable caught (of {nu}) |", "|---|---|---|"]
    lines += [f"| {r['threshold']} | {r['wrongly_blocked']} | {r['caught']} |" for r in cal["table"]]
    lines += ["", "Lowest-scoring answerable questions (these decide how high the gate can safely be):", ""]
    lines += [f"- {r['cosine']}  {r['id']}: {r['question']}" for r in cal["answerable"][:6]]
    lines += ["", "Unanswerable questions:", ""]
    lines += [f"- {r['cosine']}  {r['id']}: {r['question']}" for r in cal["unanswerable"]]
    return "\n".join(lines) + "\n"
