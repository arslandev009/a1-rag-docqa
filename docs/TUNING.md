# Hybrid retrieval tuning

Grid over keyword weight and candidate_k on 48 labelled questions (sorted by MRR, then recall@5).
**Caution:** the question set is small; prefer a simple setting over the single best row, and re-check whenever the corpus grows.

| candidate_k | keyword weight | hit@1 | hit@5 | recall@5 | MRR |
|---|---|---|---|---|---|
| 5 | 0.0 | 1.000 | 1.000 | 0.989 | 1.000 |
| 10 | 0.0 | 1.000 | 1.000 | 0.989 | 1.000 |
| 20 | 0.0 | 1.000 | 1.000 | 0.989 | 1.000 |
| 5 | 0.25 | 0.955 | 1.000 | 0.989 | 0.973 |
| 5 | 0.5 | 0.955 | 1.000 | 0.989 | 0.973 |
| 5 | 0.75 | 0.955 | 1.000 | 0.989 | 0.973 |
| 20 | 0.25 | 0.955 | 1.000 | 1.000 | 0.968 |
| 5 | 1.0 | 0.932 | 1.000 | 1.000 | 0.960 |
| 10 | 0.25 | 0.932 | 1.000 | 1.000 | 0.956 |
| 10 | 0.5 | 0.932 | 1.000 | 1.000 | 0.956 |
| 10 | 0.75 | 0.932 | 1.000 | 1.000 | 0.956 |
| 20 | 0.5 | 0.932 | 1.000 | 1.000 | 0.956 |
| 20 | 0.75 | 0.932 | 1.000 | 1.000 | 0.956 |
| 20 | 1.0 | 0.932 | 1.000 | 1.000 | 0.956 |
| 10 | 1.0 | 0.932 | 1.000 | 1.000 | 0.953 |
