# Chunk-level retrieval evaluation

44 answerable questions with gold chunks (44 answerable in total). **Gold chunks are auto-labelled from each question's `expected` text and not yet reviewed.**

`ctx_all` = share of questions whose gold chunks are all in the context the model sees; `avg chunks` = chunks sent to the model (cost).

## Current default (and the same with neighbours)

| candidate_k | keyword weight | k | neighbours | anchor_any | ctx_any | ctx_all | avg chunks | violations |
|---|---|---|---|---|---|---|---|---|
| 20 | 0.25 | 5 | 0 | 0.955 | 0.955 | 0.909 | 5.0 | 0 |
| 20 | 0.25 | 5 | 1 | 0.955 | 0.977 | 0.955 | 6.8 | 0 |

## Best 12 configurations (sorted by ctx_all, then fewest chunks)

| candidate_k | keyword weight | k | neighbours | anchor_any | ctx_any | ctx_all | avg chunks | violations |
|---|---|---|---|---|---|---|---|---|
| 20 | 0.1 | 8 | 1 | 1.000 | 1.000 | 1.000 | 9.1 | 0 |
| 10 | 0.1 | 8 | 1 | 1.000 | 1.000 | 1.000 | 9.1 | 0 |
| 10 | 0.25 | 8 | 1 | 1.000 | 1.000 | 1.000 | 9.2 | 0 |
| 10 | 0.5 | 8 | 1 | 1.000 | 1.000 | 1.000 | 9.2 | 0 |
| 20 | 0.25 | 8 | 1 | 0.977 | 1.000 | 1.000 | 9.4 | 0 |
| 20 | 0.5 | 8 | 1 | 0.977 | 1.000 | 1.000 | 9.4 | 0 |
| 20 | 0.1 | 5 | 1 | 1.000 | 1.000 | 0.977 | 6.6 | 0 |
| 10 | 0.1 | 5 | 1 | 1.000 | 1.000 | 0.977 | 6.6 | 0 |
| 10 | 0.25 | 5 | 1 | 0.977 | 1.000 | 0.977 | 6.6 | 0 |
| 10 | 0.5 | 5 | 1 | 0.977 | 1.000 | 0.977 | 6.7 | 0 |
| 10 | 0.0 | 8 | 0 | 1.000 | 1.000 | 0.977 | 8.0 | 0 |
| 10 | 0.1 | 8 | 0 | 1.000 | 1.000 | 0.977 | 8.0 | 0 |

## Questions whose gold chunks are NOT all in the context

- default, neighbours=0: q17, q44, q47, q48
- default, neighbours=1: q17, q44

## Gold chunk labels (review these; edit data/eval/gold_chunks.json)

| id | expected | gold chunks |
|---|---|---|
| q01 | Starter $19, Growth $49, Scale $129 per month | pricing-plans:1 [Plans] |
| q02 | 5 users | pricing-plans:1 [Plans] |
| q03 | 14-day trial, no card for Starter/Growth; Scale on request | pricing-plans:4 [Free trial] |
| q04 | Two months free (pay for 10) | pricing-plans:3 [Billing options] |
| q05 | No. Monthly plans are refundable only within 7 days of the first payme | refund-policy-v2:1 [Monthly subscriptions] |
| q06 | No. The 30-day guarantee on all plans was replaced; monthly is 7 days  | refund-policy-v2:0 [Refund Policy (current)], refund-policy-v2:1 [Monthly subscriptions], refund-policy-v2:2 [Annual subscriptions] |
| q07 | 5 to 10 business days (not the old 14 days) | refund-policy-v2:4 [How refunds are paid] |
| q08 | 100 requests per minute | api-rate-limits:1 [Limits per plan], api-rate-limits:4 [Batch endpoint] |
| q09 | Depends on plan: Growth 100/min, Scale 600/min, Starter has no API acc | api-rate-limits:0 [API Rate Limits], api-rate-limits:1 [Limits per plan] |
| q10 | 10 requests | api-rate-limits:1 [Limits per plan], api-rate-limits:4 [Batch endpoint] |
| q11 | Wait for Retry-After, then exponential backoff from 1 second, max 5 re | api-rate-limits:3 [Handling HTTP 429] |
| q12 | 24 hours | api-rate-limits:5 [Idempotency] |
| q13 | invoice.created, invoice.paid, invoice.overdue, payment.failed, custom | webhooks:1 [Events] |
| q14 | HMAC-SHA256 of raw body with endpoint secret, hex, compare to Ledgerly | webhooks:3 [Verifying signatures] |
| q15 | Up to 8 times over 24 hours | webhooks:2 [Delivery] |
| q16 | After 3 consecutive days of failures | webhooks:4 [Automatic disabling] |
| q17 | No. Webhooks are on Growth and Scale only | pricing-plans:1 [Plans], pricing-plans:2 [Extra users], pricing-plans:4 [Free trial], webhooks:0 [Webhooks] |
| q18 | Every 15 minutes | quickbooks-integration:1 [What syncs] |
| q19 | No, Xero is not supported | quickbooks-integration:0 [QuickBooks Integration] |
| q20 | No, only QuickBooks Online | quickbooks-integration:0 [QuickBooks Integration] |
| q21 | Every 100 days | quickbooks-integration:3 [Connection] |
| q22 | US or EU (Frankfurt) chosen at signup; cannot be changed afterwards | security-and-data:1 [Data location] |
| q23 | No. SOC 2 Type II audit in progress, report expected Q1 2027 | security-and-data:3 [Compliance] |
| q24 | 35 days | security-and-data:2 [Backups and retention] |
| q25 | Retained 90 days, then permanently deleted | security-and-data:2 [Backups and retention] |
| q26 | SAML SSO on Scale only | pricing-plans:1 [Plans], security-and-data:4 [Access control] |
| q27 | Complete the 3-D Secure authentication step in the payment link | failed-payments-troubleshooting:0 [Common error codes] |
| q28 | Day 1, day 3 and day 7, then the invoice is marked overdue | failed-payments-troubleshooting:1 [Automatic retries and reminders] |
| q29 | $15 per dispute | failed-payments-troubleshooting:2 [Payouts and disputes] |
| q30 | First response within 1 hour, around the clock | support-and-sla:0 [Support channels] |
| q31 | 99.9% monthly; 10% credit if 99.0-99.9%, 25% below 99.0%; claim within | support-and-sla:1 [Uptime] |
| q34 | Must not reveal internal goodwill rules; point to public policy and bi | refund-policy-v2:4 [How refunds are paid] |
| q37 | No for renewals; refund only within 7 days of the first payment | refund-policy-v2:1 [Monthly subscriptions] |
| q38 | Honour Retry-After, exponential backoff from 1 second, max 5 retries | api-rate-limits:3 [Handling HTTP 429] |
| q39 | Data is encrypted at rest with AES-256 | security-and-data:0 [Encryption] |
| q40 | Choose the EU (Frankfurt) region at signup; cannot change later | security-and-data:1 [Data location] |
| q41 | authentication_required: customer completes 3-D Secure in the payment  | failed-payments-troubleshooting:0 [Common error codes] |
| q42 | Verify the HMAC-SHA256 Ledgerly-Signature; reject timestamps older tha | webhooks:3 [Verifying signatures] |
| q43 | Synced every 15 minutes | quickbooks-integration:1 [What syncs] |
| q44 | Starter, $19/month, up to 50 invoices per month | pricing-plans:1 [Plans] |
| q45 | Scale gets a 10% or 25% service credit; Growth has no credits; Starter | support-and-sla:1 [Uptime] |
| q46 | $15 per dispute | failed-payments-troubleshooting:2 [Payouts and disputes] |
| q47 | SAML single sign-on, Scale plan only | pricing-plans:1 [Plans], security-and-data:4 [Access control] |
| q48 | 14-day trial, no card for Starter/Growth | pricing-plans:4 [Free trial] |
