# Provider comparison (end-to-end answers)

Same retrieval, same prompt, same labelled questions; only the answering model changes. Judge = a different provider. Costs use list prices from `prices.json` (free tiers bill $0).

| Provider / model | answer accuracy | false abstain | abstention acc. | probe pass | citation hit | grounded | leaks | p50 s | p95 s | avg tokens in/out | $ per 1k answers |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini/gemini-3.5-flash-lite | 100% | 0% | 100% | 100% | 100% | 100% | 0 | 1.76 | 2.2 | 775/77 | 0.426 |

## Judge cross-check

Every answer is graded by each judge. The headline accuracy above uses a judge from a different provider than the answerer. If a judge scores its own provider's answers much higher than the other judge does, that is self-preference bias.

| Answerer | judge | answer accuracy per this judge | judges agree on |
|---|---|---|---|
| gemini/gemini-3.5-flash-lite | gemini | 100% | 100% |
| gemini/gemini-3.5-flash-lite | groq | 100% | 100% |

## Failures: gemini/gemini-3.5-flash-lite

- none
