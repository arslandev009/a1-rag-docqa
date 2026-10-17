---
title: Webhooks
doc_type: api
version: "2026-07"
status: current
audience: public
effective_date: 2026-07-01
source_url: https://help.ledgerly.example/api/webhooks
---
# Webhooks

Webhooks are available on the Growth and Scale plans.

## Events

Ledgerly sends these events: invoice.created, invoice.paid, invoice.overdue, payment.failed and customer.updated.

## Delivery

Endpoints must use HTTPS. Ledgerly sends a POST request with a JSON body and waits up to 10 seconds for a 2xx response. Failed deliveries are retried up to 8 times over 24 hours.

## Verifying signatures

Each request carries a Ledgerly-Signature header and a Ledgerly-Timestamp header. Compute an HMAC-SHA256 of the raw request body with your endpoint secret, hex-encode it and compare it to the header. Reject requests whose timestamp is older than 5 minutes.

## Automatic disabling

If an endpoint fails for 3 consecutive days, Ledgerly disables it and emails the account owner. Re-enable it from Settings, Webhooks.
