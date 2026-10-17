"""Command line: docqa ingest | stats | search | ask | eval-retrieval | tune-retrieval | calibrate | eval-answers | compare | diagnose"""
from __future__ import annotations

import argparse
import json
import os
import re

from .config import ROOT, load_settings
from .embedder import Embedder
from . import eval_answers
from .eval_retrieval import (calibrate, calibration_markdown, embed_queries, evaluate, load_questions, sweep,
                             sweep_markdown, to_markdown)
from .ingest import ingest
from .loader import load_corpus
from .retrieval import MODES, Retriever
from .store import Store


def _print_results(title: str, items) -> None:
    print(f"\n{title}")
    if not items:
        print("  (no results)")
    for r in items:
        h = r.hit
        flag = "" if (h.status, h.audience) == ("current", "public") else f"  [{h.audience}/{h.status}]"
        ranks = f"v{r.vector_rank or '-'}/k{r.keyword_rank or '-'}"
        snippet = h.content.split("\n\n", 1)[-1].replace("\n", " ")[:80]
        print(f"  {r.score:7.4f}  {ranks:<7} {h.doc_id:<30} {h.heading_path.split(' > ')[-1][:24]:<24} {snippet}{flag}")


def _print_answer(res) -> None:
    print(f"\nQ: {res.question}\n\nA: {res.answer}")
    if res.citations:
        print("\nSources:")
        for c in res.citations:
            print(f"  [{c.id}] {c.title} > {c.heading}   {c.url}")
    cos = f"{res.top_vector_score:.3f}" if res.top_vector_score is not None else "-"
    print(f"\nabstained={res.abstained} reason={res.reason} | top cosine={cos} | llm_calls={res.llm_calls} | "
          f"{res.provider}/{res.model} | tokens {res.prompt_tokens}+{res.completion_tokens} | "
          f"{res.latency_s:.2f}s | ${res.cost_usd:.5f}")
    if res.error:
        print(f"error detail: {res.error}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="docqa")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest", help="chunk, embed and store the corpus (only changed chunks are re-embedded)")
    pi.add_argument("--reset", action="store_true", help="drop and rebuild all tables (needed when the embedding model changes)")
    pi.add_argument("--prune", action="store_true", help="delete documents that no longer exist in the corpus folder")
    sub.add_parser("stats", help="show what is in the database")
    ps = sub.add_parser("search", help="retrieve chunks for a question (debugging)")
    ps.add_argument("query")
    ps.add_argument("-k", type=int, default=5)
    ps.add_argument("--mode", choices=MODES, default="hybrid")
    ps.add_argument("--include-all", action="store_true", help="DEBUG: also search deprecated and internal documents")
    pa = sub.add_parser("ask", help="answer a question with citations (retrieval + LLM)")
    pa.add_argument("question")
    pa.add_argument("--provider", default=None, help="openai | groq | gemini | local (default: LLM_PROVIDER)")
    pa.add_argument("--model", default=None, help="override the model id for this call")
    pe = sub.add_parser("eval-retrieval", help="hit@k / MRR of vector, keyword and hybrid retrieval on the labelled questions")
    pe.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    sub.add_parser("tune-retrieval", help="grid-search the hybrid knobs (keyword weight, candidate_k); no API cost")
    sub.add_parser("calibrate", help="how well can a similarity gate separate answerable from unanswerable questions (no API cost)")
    pv = sub.add_parser("eval-answers", help="run all labelled questions end to end, judge them, summarise")
    pv.add_argument("--provider", default=None)
    pv.add_argument("--model", default=None)
    pv.add_argument("--judges", default="groq,gemini", help="comma separated judge providers; each grades every answer")
    pv.add_argument("--sleep", type=float, default=2.0, help="seconds between calls (free-tier rate limits)")
    pv.add_argument("--limit", type=int, default=None)
    pv.add_argument("--only", default=None, help="comma separated question ids, e.g. q05,q06")
    pv.add_argument("--fresh", action="store_true", help="ignore saved answers for this provider/model and start again")
    sub.add_parser("compare", help="build docs/RESULTS.md from all saved eval-answers summaries")
    pc = sub.add_parser("eval-chunks", help="chunk-level retrieval eval + context sweep (gold chunks auto-labelled, see data/eval/gold_chunks.json)")
    pc.add_argument("--relabel", action="store_true", help="regenerate data/eval/gold_chunks.json (overwrites hand edits)")
    sub.add_parser("diagnose", help="for every failed question show what was retrieved and which gold chunks were missing")
    args = p.parse_args(argv)

    s = load_settings()
    if not os.environ.get("LLM_PRICES_FILE"):
        os.environ["LLM_PRICES_FILE"] = str(ROOT / "prices.json")
    if args.cmd == "compare":
        sums = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((ROOT / "results").glob("summary_*.json"))]
        if not sums:
            print("No summaries yet. Run: docqa eval-answers --provider gemini")
            return 1
        (ROOT / "docs").mkdir(exist_ok=True)
        md = eval_answers.compare_markdown(sums)
        (ROOT / "docs" / "RESULTS.md").write_text(md, encoding="utf-8")
        print(md)
        print("saved: docs/RESULTS.md")
        return 0
    store = Store(s.database_url, s.db_schema)
    try:
        if args.cmd == "stats":
            st = store.stats()
            if not st["meta"]:
                print("Database is empty. Run: docqa ingest")
                return 0
            print(f"embedding: {st['meta']['embed_model']} ({st['meta']['embed_dim']} dims)")
            print(f"documents: {st['documents']}   chunks: {st['chunks']}")
            for k, n in st["by_status"].items():
                print(f"  {k:<22} {n}")
            return 0

        emb = Embedder(s.embed_provider, s.embed_model, batch_size=s.embed_batch_size)
        if args.cmd == "ingest":
            dim = emb.probe_dim()
            store.init_schema(dim, emb.model, reset=args.reset)
            rep = ingest(load_corpus(s.corpus_dir), emb, store, max_chars=s.chunk_max_chars,
                         overlap_sentences=s.chunk_overlap_sentences, prune=args.prune)
            print(f"docs={rep.docs} chunks={rep.chunks} embedded={rep.embedded} unchanged={rep.unchanged} "
                  f"deleted_chunks={rep.deleted_chunks} deleted_docs={rep.deleted_docs} dim={dim} {rep.seconds:.1f}s")
            return 0

        if not store.load_meta():
            print("Database is empty. Run: docqa ingest")
            return 1
        retriever = Retriever(store, emb, candidate_k=s.candidate_k, keyword_weight=s.keyword_weight)

        if args.cmd == "search":
            aud = ("public", "internal") if args.include_all else ("public",)
            sts = ("current", "deprecated") if args.include_all else ("current",)
            if args.include_all:
                print("WARNING: --include-all is a debugging switch; the assistant never uses it.")
            res = retriever.retrieve(args.query, k=args.k, mode=args.mode, audiences=aud, statuses=sts)
            _print_results(f"{args.mode} retrieval", res.items)
            top = res.top_vector_score
            print(f"\ntop vector cosine: {top:.3f}" if top is not None else "")
            return 0

        if args.cmd == "ask":
            from llm_client import LLM

            from .generate import Assistant

            assistant = Assistant(retriever, LLM(args.provider, model=args.model), k=s.answer_k,
                                  min_cosine=s.abstain_min_cosine, max_tokens=s.answer_max_tokens,
                                  temperature=s.answer_temperature)
            _print_answer(assistant.ask(args.question))
            return 0

        questions = load_questions(s.eval_path)
        vecs = embed_queries(emb, questions, ROOT / ".cache" / "query_embeddings.json")
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "docs").mkdir(exist_ok=True)

        if args.cmd == "diagnose":
            for f in sorted((ROOT / "results").glob("answers_*.jsonl")):
                recs = eval_answers.load_records(f)
                summ = eval_answers.summarise(recs, {})
                print(f"=== {f.stem.replace('answers_', '')}: {len(summ['failures'])} failures ===\n")
                print(eval_answers.diagnose(store, recs, summ["failures"]))
            return 0

        if args.cmd == "calibrate":
            cal = calibrate(retriever, questions, vecs)
            md = calibration_markdown(cal)
            (ROOT / "docs" / "CALIBRATION.md").write_text(md, encoding="utf-8")
            print(md)
            print("saved: docs/CALIBRATION.md")
            return 0

        if args.cmd == "eval-answers":
            from llm_client import LLM

            from .generate import Assistant

            llm = LLM(args.provider, model=args.model, max_retries=6)  # long backoff rides out free-tier rate limits
            judges = {n.strip(): LLM(n.strip(), max_retries=6) for n in args.judges.split(",") if n.strip()}
            assistant = Assistant(retriever, llm, k=s.answer_k, min_cosine=s.abstain_min_cosine,
                                  max_tokens=s.answer_max_tokens, temperature=s.answer_temperature)
            tag = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{llm.provider}_{llm.model}")
            print(f"answering with {llm.provider}/{llm.model}; judges: " + ", ".join(f"{n}/{j.model}" for n, j in judges.items()))
            only = set(args.only.split(",")) if args.only else None
            summary = eval_answers.run(assistant, judges, questions, ROOT / "results", tag, sleep=args.sleep,
                                       only=only, limit=args.limit, fresh=args.fresh)
            (ROOT / "docs").mkdir(exist_ok=True)
            print("\n" + eval_answers.compare_markdown([summary]))
            print(f"saved: results/answers_{tag}.jsonl, results/summary_{tag}.json (run `docqa compare` for the table)")
            return 0

        if args.cmd == "eval-chunks":
            from . import eval_chunks

            md = eval_chunks.run(store, emb, questions, vecs, ROOT / "data" / "eval" / "gold_chunks.json", relabel=args.relabel)
            (ROOT / "docs" / "CHUNK_EVAL.md").write_text(md, encoding="utf-8")
            print(md)
            print("saved: docs/CHUNK_EVAL.md and data/eval/gold_chunks.json")
            return 0

        if args.cmd == "tune-retrieval":
            rows = sweep(store, emb, questions, vecs)
            md = sweep_markdown(rows, len(questions))
            (ROOT / "docs" / "TUNING.md").write_text(md, encoding="utf-8")
            print(md)
            print("saved: docs/TUNING.md")
            return 0

        report = evaluate(retriever, questions, vecs, modes=args.modes)
        (ROOT / "results" / "retrieval_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        md = to_markdown(report)
        (ROOT / "docs" / "RETRIEVAL_RESULTS.md").write_text(md, encoding="utf-8")
        print(md)
        print("saved: results/retrieval_eval.json and docs/RETRIEVAL_RESULTS.md")
        return 0
    finally:
        store.close()
