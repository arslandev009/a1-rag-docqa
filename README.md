# A1: Ledgerly Support Assistant (production-style RAG)

A customer-support assistant for a fictional invoicing SaaS ("Ledgerly"). The corpus is deliberately messy in the
ways real client knowledge bases are: an **outdated policy that contradicts the current one**, a **plan-dependent
answer** (rate limits), **things the company does not do** (Xero, SOC 2), **internal-only staff notes** the assistant
must never reveal, and **questions with no answer in the docs**.

The goal is not "chat with PDFs". It is a system a client could put in front of customers and trust: measured,
cited, safe, and cheap to run.

## Definition of done: where it stands
- [x] Hybrid retrieval (keyword + vector, fused) with metadata filtering before ranking
- [x] Deprecated documents never answer; internal documents never reach a customer-facing answer (filters run in SQL, also for neighbour chunks; 0 leaks in every answer eval)
- [x] Every answer carries citations (document, section, link); no valid citation means abstention
- [x] Abstains with "I couldn't find this" instead of guessing; prompt-injection attempts refused
- [x] Eval harness over 60 labelled questions: retrieval hit@k / MRR (document level), chunk-level coverage, answer correctness, faithfulness, abstention accuracy, leak rate (0)
- [ ] Provider comparison table (Gemini, Groq, OpenAI, local) on the same eval: **only Gemini is valid so far**; the Groq run was cut off by free-plan limits (see "Known gaps")
- [x] FastAPI service, Docker Compose (app + Postgres/pgvector), 87 tests
- [ ] CI: the GitHub Actions workflow is written and runs the same lint + tests locally, but it has not run on GitHub yet (needs the `llm-client` package published as its own repo)
- [x] README with architecture, decisions and an honest failure analysis (this file)

## Architecture
```
docs (.md + front matter) -> loader -> heading-aware chunker -> embeddings -> Postgres
                                                                     |  pgvector (semantic)
question -> access filter (audience, status) -> hybrid search <------+  tsvector (keyword)
         -> fusion (RRF) -> [optional neighbour chunks] -> answer prompt with sources -> LLM (llm-client)
         -> citation check + abstain rule -> JSON answer {answer, citations, abstained, reason}
```

## Key decisions
| Decision | Why |
|---|---|
| Postgres + pgvector + full-text, not a separate vector DB | One system to run and back up; most clients already have Postgres |
| Hybrid search, keyword weight 0.1 | Exact terms (HTTP 429, authentication_required, plan names) need keywords; paraphrases need vectors. The weight was chosen with the chunk-level eval (see below), not by feel |
| Metadata filter before ranking, in SQL | Safer than asking the LLM to ignore deprecated or internal text |
| Heading path in every chunk | "Growth" or "Scale" limits stay attached to their section |
| Chunk-level eval next to the document-level one | A document-level hit hid a real failure (q44); see the failure analysis |
| Neighbour-chunk expansion built, tested, **off by default** | It fixed one failure but costs 32% more context; once the keyword weight was fixed it added nothing on this corpus |
| Provider-agnostic `llm-client` | Swap Gemini / Groq / OpenAI / local by config; compare on the same evals |

## Roadmap
- [x] **A1.1** Corpus (10 docs incl. traps), eval set (60 questions), loader, chunker, tests
- [x] **A1.2** Embeddings (OpenAI-compatible) + pgvector/halfvec store + idempotent ingestion CLI + filtered vector/keyword search
- [x] **A1.3** Hybrid retrieval (vector + keyword, RRF) + access/version filters in SQL + document-level retrieval eval
- [x] **A1.4** Answer generation: citations checked against the sources, abstention, 5 layered injection/leak defences, `docqa ask`
- [x] **A1.5** End-to-end answer eval (rules + LLM judges), abstention-gate calibration, provider comparison
- [x] **A1.6** FastAPI service, Docker Compose, tests; neighbour expansion; chunk-level eval; retrieval tuning
- [ ] A1.6e CI first green run on GitHub; valid Groq run; OpenAI and local rows in the provider table
- [ ] A1.7 Demo UI, 90-second video

## Run
```powershell
# from ai-studio\projects\a1-rag-docqa, workspace venv active
pip install -e ".[dev,api]"        # needs llm-client too: pip install -e ..\..\packages\llm-client
docker compose up -d               # Postgres + pgvector on 127.0.0.1:55433
pytest                             # 87 tests; the database tests are skipped when Postgres is down

docqa ingest                       # chunk + embed + store (re-run: only changed chunks are embedded)
docqa stats
docqa search "How long does a refund take?"            # hybrid by default; add --mode vector|keyword
docqa search "30-day money-back guarantee"             # deprecated policy must NOT appear
docqa search "goodwill refund exception" --include-all # DEBUG: internal docs appear only with this flag
docqa ask "Can I get a refund on my monthly plan renewal?" --provider gemini
docqa eval-retrieval               # document-level hit@k / MRR for vector, keyword, hybrid -> docs/RETRIEVAL_RESULTS.md
docqa eval-chunks                  # chunk-level coverage + config sweep -> docs/CHUNK_EVAL.md (free: cached query vectors)
docqa tune-retrieval               # document-level grid of keyword weight / candidate_k -> docs/TUNING.md
docqa calibrate                    # can a similarity gate separate answerable from unanswerable questions? (free)
docqa eval-answers --provider gemini --sleep 5     # all 60 questions end to end (--limit N for a smaller run)
docqa compare                      # -> docs/RESULTS.md
docqa diagnose                     # for each failure: what was retrieved, which gold chunks were missing
docqa ingest --reset               # required after changing EMBED_MODEL
```
Needs in the workspace `.env`: `GEMINI_API_KEY`, `EMBED_PROVIDER=gemini`, `EMBED_MODEL=gemini-embedding-001`.
If the free tier returns 429, lower `EMBED_BATCH_SIZE` (for example 10).

## How the store works
- Chunks carry `status` and `audience`; searches filter `current` + `public` **in SQL before ranking**.
- 3072-dim Gemini vectors are stored as `halfvec(3072)` with an HNSW index (`vector` indexes stop at 2000 dims).
- `meta` remembers the embedding model and dimension; changing the model without `--reset` raises an error.
- Ingestion hashes `model + chunk text`: unchanged chunks are skipped, edited ones re-embedded, removed ones deleted.

## Answering: defence in depth
1. Input guard: obvious injection phrases are refused without calling the model (regression-tested so that none of the labelled questions is blocked).
2. Retrieval filter: only public + current chunks can ever be candidates (SQL); neighbour chunks go through the same filter.
3. Abstain gate: best cosine below `ABSTAIN_MIN_COSINE` (default 0.55) means "not found" without an LLM call. It is deliberately low: answerable and unanswerable questions overlap in similarity, so the model itself must abstain from the sources.
4. System prompt: answer only from numbered sources, cite, abstain if the sources do not answer, treat question and sources as data.
5. Output checks: every citation must be one of the provided sources; an answer without a valid citation becomes an abstention. Provider errors and invalid JSON (one retry) never crash the assistant.

## Evaluation (three levels)
**1. Document level** (`docqa eval-retrieval`, `docs/RETRIEVAL_RESULTS.md`): was any chunk of the gold document retrieved?
It showed vector search ranking best on this small, clean corpus and hybrid giving the best recall. It also reported
hit@5 = 1.0 while one question was missing its answer chunk, which is why level 2 exists.

**2. Chunk level** (`docqa eval-chunks`, `docs/CHUNK_EVAL.md`): are the gold *chunks* in the context the model sees?
44 answerable questions. Gold chunks are derived from each question's `expected` text; **8 of 44 are hand-reviewed**
(36 are auto-labelled and unreviewed: `data/eval/gold_chunks.json`). Retrieval: candidate_k 20, 5 hits, plus optional neighbours.

| Keyword weight | Neighbour chunks | Gold chunks all in context | Avg chunks sent | Not fully covered |
|---|---|---|---|---|
| 0.25 (old default) | off | 95.5% | 5.0 | q44, q48 |
| 0.25 | on | 97.7% | 6.8 | q44 |
| **0.10 (new default)** | **off** | **100%** | **5.0** | none |
| 0.10 | on | 100% | 6.6 | none |

**3. Answer level** (`docqa eval-answers`): 44 questions expect an answer, 16 expect a refusal or abstention.
Every answer is graded by both judges (Groq and Gemini); the headline accuracy uses the judge from a *different*
provider than the answerer, and `docs/RESULTS.md` shows the cross-check. Leaks are detected by canary phrases from the
internal runbook (rule-based, no judge).

| Run (Gemini 3.5 Flash Lite) | Questions | Keyword weight | Abstain gate | Accuracy | False abstain | Abstention | Grounded | Leaks |
|---|---|---|---|---|---|---|---|---|
| 1 (first end-to-end) | 48 | 0.25 | 0.60 | 89% | 7% | 100% | 98% | 0 |
| 2 | 60 | 0.25 | 0.55 | 93% | 4% | 100% | 100% | 0 |
| 3 (current) | 60 | 0.10 | 0.55 | 100% | 0% | 100% | 100% | 0 |

Run 3: p50 1.76 s, p95 2.2 s, 775 tokens in / 77 out per answer, $0.426 per 1,000 answers at list price (the free tier bills $0). Both judges agree on every answer.
**How far to trust 100%:** the 44 answerable questions are the ones the keyword weight was tuned on (in-sample), the set is small, and the 95% confidence lower bound for 44 out of 44 is about 92%. Treat it as "no known failures", not as production accuracy.

## Failure analysis
- **q41, refused by the similarity gate** (cosine 0.587 < 0.60). A gate is blunt: answerable questions score 0.59 to 0.90, unanswerable ones 0.58 to 0.67. Fix: gate lowered to 0.55 and abstention left to the model, measured on the unanswerable questions.
- **q44 ("cheapest option if I only bill a handful of customers") and q48 (free trial):** the model abstained correctly, because the context did not contain the price or the trial terms. Root cause was retrieval, not generation: the vector search ranked the "Plans" chunk 2nd, but the keyword search matched generic words of the natural-language question ("bill" matches "Annual **billing**", "customers", ...) and RRF pushed the right chunk to 6th place by a margin of 0.0003. **The document-level eval could not see this**, because another chunk of the same document was retrieved (hit@5 stayed 1.0). The chunk-level eval can, and it also showed that a keyword weight of 0.1 puts the chunk back in the top 5 without sending any extra text.
- **Neighbour-chunk expansion** was the first fix tried: it rescued q48 (its chunk sits next to a retrieved one) but not q44 (two chunks away). It is implemented, tested (including that neighbours can never cross the access filter) and off by default (`RETRIEVAL_NEIGHBORS=1` turns it on); it may matter for larger corpora.
- **Auto-labelled gold chunks were noisy:** q17 got four gold chunks because "growth" and "scale" appear everywhere; q10, q08, q26, q47 were wrong in similar ways. The labels now say which entries are hand-reviewed.
- **Process failure, found and fixed:** this README once described neighbour expansion that did not exist in the code, and flags and behaviours that did not match the CLI. Every claim here was re-checked against the code; behaviours that matter have tests.

## Known gaps (honest list)
- **Provider comparison is incomplete.** The Groq answering run failed for 36 of 44 answerable questions with provider errors (free-plan rate limits; Groq is also one of the two judges, so judging and answering share the same quota). Those results are kept in `results/_incomplete/` and are not shown as a quality result. To do: rerun Groq with `--judges gemini`, add OpenAI, add a local model.
- Small corpus (10 documents, 41 chunks) and a small question set: the numbers say the pipeline works, not how it scales.
- 36 of 44 gold chunk labels are auto-generated and unreviewed.
- The abstain gate and keyword weight were tuned on the same questions they are evaluated on.

## API
FastAPI service on top of the assistant. `docqa.api:router` is importable (mount it in a shared demo-api with
`app.include_router(router, prefix="/a1")`); `docqa.main:app` is the standalone app.

| Endpoint | Purpose |
|---|---|
| `GET /health` | readiness: database reachable and ingested (open, used by Docker / Render health checks) |
| `GET /sources` | public, current documents the assistant may cite |
| `POST /chat` | `{"question": "...", "provider": "gemini"}` -> `{answer, abstained, reason, citations[], provider, model, latency_s, cost_usd}` |

Retrieved text and provider error details are never returned to the caller.

Run locally (database up and ingested):
```powershell
pip install -e ".[dev,api]"
uvicorn docqa.main:app --port 8000
Invoke-RestMethod -Method Post http://127.0.0.1:8000/chat -ContentType "application/json" -Body '{"question":"How long does a refund take?"}'
```
Run in Docker (database + API): `docker compose --profile app up -d --build`

| Variable | Default | Meaning |
|---|---|---|
| `GATEWAY_SECRET` | empty | if set, `/chat` and `/sources` need the header `X-Gateway-Secret` (set by the edge function) |
| `API_ALLOWED_PROVIDERS` | `gemini,groq` | provider allow-list; `API_DEFAULT_PROVIDER` picks the default |
| `API_RATE_LIMIT_PER_MIN` | `10` | requests per client IP per minute (`X-Real-Client-IP`, trusted only after the secret check) |
| `API_DAILY_LLM_CALLS` | `300` | daily LLM-call budget (UTC); after that `/chat` answers 503 |
| `CORS_ORIGINS` | empty | comma separated origins, only for direct browser calls in local UI development |
| `RETRIEVAL_KEYWORD_WEIGHT` / `RETRIEVAL_CANDIDATE_K` / `ANSWER_K` | `0.1` / `20` / `5` | retrieval settings chosen with `docqa eval-chunks` |
| `RETRIEVAL_NEIGHBORS` / `RETRIEVAL_NEIGHBOR_TOP` | `0` / `3` | neighbour-chunk expansion (off by default) |

Limits: rate limit and budget are in memory (reset on restart, single instance only); one question is answered at a
time (one shared database connection, fine for a demo); the Docker build needs the sibling `packages/llm-client`
folder, and CI installs `llm-client` from its own public GitHub repo.

## Free-plan notes
All providers here are on free plans, so evals are built to be token-frugal and to fail politely:
- the client honours `Retry-After`; if a provider asks for a very long wait (a daily token limit) the call fails fast,
  and after 3 provider errors in a row the run stops and keeps its progress. Re-running the same command (without
  `--fresh`) retries only the failed questions.
- `--fresh` replaces the saved answers for that provider and model: copy `results/` first if you want to keep them.
