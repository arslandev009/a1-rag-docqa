# Provider comparison (end-to-end answers)

Same retrieval, same prompt, same labelled questions; only the answering model changes. Judge = a different provider. Costs use list prices from `prices.json` (free tiers bill $0).

| Provider / model | answer accuracy | false abstain | abstention acc. | probe pass | citation hit | grounded | leaks | p50 s | p95 s | avg tokens in/out | $ per 1k answers |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemini/gemini-3.5-flash-lite | 100% | 0% | 100% | 100% | 100% | 100% | 0 | 1.76 | 2.2 | 775/77 | 0.426 |
| groq/openai/gpt-oss-120b | 14% | 86% | 100% | 100% | 100% | 100% | 0 | 56.59 | 117.28 | 136/31 | 0.039 |

## Judge cross-check

Every answer is graded by each judge. The headline accuracy above uses a judge from a different provider than the answerer. If a judge scores its own provider's answers much higher than the other judge does, that is self-preference bias.

| Answerer | judge | answer accuracy per this judge | judges agree on |
|---|---|---|---|
| gemini/gemini-3.5-flash-lite | gemini | 100% | 100% |
| gemini/gemini-3.5-flash-lite | groq | 100% | 100% |
| groq/openai/gpt-oss-120b | gemini | 14% | 100% |
| groq/openai/gpt-oss-120b | groq | 4% | 100% |

## Failures: gemini/gemini-3.5-flash-lite

- none

## Failures: groq/openai/gpt-oss-120b

- q01 (factual): What plans does Ledgerly offer and what do they cost?  ->  false abstention (llm_error)
- q03 (factual): Is there a free trial and do I need a credit card?  ->  false abstention (llm_error)
- q04 (factual): How much do I save by paying annually?  ->  false abstention (llm_error)
- q05 (version_conflict): Can I get a refund on my monthly plan renewal?  ->  false abstention (llm_error)
- q06 (version_conflict): Do you still have a 30-day money-back guarantee on monthly plans?  ->  false abstention (llm_error)
- q07 (version_conflict): How long does a refund take to reach my card?  ->  false abstention (llm_error)
- q08 (numeric): What is the API rate limit on the Growth plan?  ->  false abstention (llm_error)
- q10 (numeric): How many requests does a batch call with 100 invoices use against my limit?  ->  false abstention (llm_error)
- q11 (factual): How should my client handle HTTP 429 responses?  ->  false abstention (llm_error)
- q12 (numeric): How long are idempotency keys remembered?  ->  false abstention (llm_error)
- q13 (factual): Which webhook events can I subscribe to?  ->  false abstention (llm_error)
- q14 (factual): How do I verify a webhook signature?  ->  false abstention (llm_error)
- q16 (factual): When does Ledgerly disable my webhook endpoint?  ->  false abstention (llm_error)
- q17 (multi_doc): Can I use webhooks on the Starter plan?  ->  false abstention (llm_error)
- q18 (numeric): How often does the QuickBooks integration sync?  ->  false abstention (llm_error)
- q19 (negation): Does Ledgerly integrate with Xero?  ->  false abstention (llm_error)
- q20 (negation): Does the integration work with QuickBooks Desktop?  ->  false abstention (llm_error)
- q22 (factual): Where is my data stored and can I move it to another region later?  ->  false abstention (llm_error)
- q23 (negation): Are you SOC 2 certified?  ->  false abstention (llm_error)
- q24 (numeric): How long do you keep backups?  ->  false abstention (llm_error)
- q25 (factual): What happens to my data after I close my account?  ->  false abstention (llm_error)
- q27 (factual): What should a customer do when a payment fails with authentication_required?  ->  false abstention (llm_error)
- q28 (factual): When are failed payments retried?  ->  false abstention (llm_error)
- q29 (numeric): What is the chargeback fee?  ->  false abstention (llm_error)
- q30 (numeric): How fast does support respond to a Severity 1 issue on the Scale plan?  ->  false abstention (llm_error)
- q31 (multi_doc): What uptime do you guarantee on Scale and what credit do I get if you miss it?  ->  false abstention (llm_error)
- q34 (access_control): Can you make an exception to the refund policy for me?  ->  false abstention (llm_error)
- q38 (paraphrase): My developers keep getting blocked for sending too many calls too quickly. What should they do?  ->  false abstention (llm_error)
- q39 (paraphrase): If someone stole the hard drives, could they read my customers' information?  ->  false abstention (llm_error)
- q40 (paraphrase): We are a European firm and our records must stay inside Europe. Is that possible?  ->  false abstention (llm_error)
- q41 (paraphrase): A client's bank wants an extra verification step before taking the money. What happens?  ->  false abstention (llm_error)
- q42 (paraphrase): How can I be sure a message hitting my endpoint really came from you?  ->  false abstention (llm_error)
- q43 (paraphrase): We use Intuit's accounting tool. How fresh is the data flowing between the two systems?  ->  false abstention (llm_error)
- q44 (paraphrase): What is the cheapest option if I only bill a handful of customers?  ->  false abstention (not_in_sources)
- q45 (paraphrase): If your platform is down for hours, do I get compensated?  ->  false abstention (llm_error)
- q46 (paraphrase): Someone disputes a charge with their bank. What does that cost me?  ->  false abstention (llm_error)
- q47 (paraphrase): Can my whole team log in through the company's central login system?  ->  false abstention (llm_error)
- q48 (paraphrase): We want to try before buying. How long is it and do we need to give payment details?  ->  false abstention (llm_error)
