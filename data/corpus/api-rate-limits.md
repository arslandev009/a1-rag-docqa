---
title: API Rate Limits
doc_type: api
version: "2026-07"
status: current
audience: public
effective_date: 2026-07-01
source_url: https://help.ledgerly.example/api/rate-limits
---
# API rate limits

API access is available on the Growth and Scale plans. Starter accounts do not have API access.

## Limits per plan

- Growth: 100 requests per minute.
- Scale: 600 requests per minute.
- Bursts of up to twice the limit are allowed for 10 seconds.

## Response headers

Every response includes X-RateLimit-Limit, X-RateLimit-Remaining and X-RateLimit-Reset (a Unix timestamp in seconds).

## Handling HTTP 429

When you exceed the limit the API returns HTTP 429 with a Retry-After header in seconds. Wait at least that long, then retry with exponential backoff starting at 1 second, with at most 5 retries.

## Batch endpoint

POST /invoices/batch accepts up to 100 invoices per call. One batch call counts as 10 requests against your limit.

## Idempotency

Send an Idempotency-Key header on POST requests to make retries safe. Keys are remembered for 24 hours.
