# Retrieval evaluation

48 labelled questions; metrics are averaged over those with gold source documents.

| Mode | hit@1 | hit@3 | hit@5 | recall@5 | MRR | policy violations |
|---|---|---|---|---|---|---|
| vector | 1.000 | 1.000 | 1.000 | 0.989 | 1.000 | 0 |
| keyword | 0.750 | 0.886 | 0.909 | 0.909 | 0.824 | 0 |
| hybrid | 0.909 | 0.977 | 1.000 | 1.000 | 0.941 | 0 |

## By question type: vector

| Type | n | hit@5 | MRR |
|---|---|---|---|
| access_control | 1 | 1.000 | 1.000 |
| ambiguity | 1 | 1.000 | 1.000 |
| factual | 11 | 1.000 | 1.000 |
| multi_doc | 3 | 1.000 | 1.000 |
| negation | 3 | 1.000 | 1.000 |
| numeric | 10 | 1.000 | 1.000 |
| paraphrase | 12 | 1.000 | 1.000 |
| version_conflict | 3 | 1.000 | 1.000 |

Top-1 vector cosine, used to choose the abstention threshold in A1.4:

| Questions | min | median | max |
|---|---|---|---|
| answerable | 0.587 | 0.751 | 0.898 |
| unanswerable | 0.579 | 0.586 | 0.665 |

## By question type: keyword

| Type | n | hit@5 | MRR |
|---|---|---|---|
| access_control | 1 | 1.000 | 1.000 |
| ambiguity | 1 | 1.000 | 1.000 |
| factual | 11 | 1.000 | 0.909 |
| multi_doc | 3 | 1.000 | 1.000 |
| negation | 3 | 1.000 | 0.833 |
| numeric | 10 | 1.000 | 0.950 |
| paraphrase | 12 | 0.667 | 0.562 |
| version_conflict | 3 | 1.000 | 0.833 |

Misses (gold document not in top 5):

- q39: If someone stole the hard drives, could they read my customers' information? (gold: security-and-data)
- q43: We use Intuit's accounting tool. How fresh is the data flowing between the two systems? (gold: quickbooks-integration)
- q47: Can my whole team log in through the company's central login system? (gold: security-and-data, pricing-plans)
- q48: We want to try before buying. How long is it and do we need to give payment details? (gold: pricing-plans)

## By question type: hybrid

| Type | n | hit@5 | MRR |
|---|---|---|---|
| access_control | 1 | 1.000 | 1.000 |
| ambiguity | 1 | 1.000 | 1.000 |
| factual | 11 | 1.000 | 1.000 |
| multi_doc | 3 | 1.000 | 1.000 |
| negation | 3 | 1.000 | 0.833 |
| numeric | 10 | 1.000 | 1.000 |
| paraphrase | 12 | 1.000 | 0.826 |
| version_conflict | 3 | 1.000 | 1.000 |

Top-1 vector cosine, used to choose the abstention threshold in A1.4:

| Questions | min | median | max |
|---|---|---|---|
| answerable | 0.587 | 0.751 | 0.898 |
| unanswerable | 0.579 | 0.586 | 0.665 |
