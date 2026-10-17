---
title: QuickBooks Integration
doc_type: integration
version: "2026-07"
status: current
audience: public
effective_date: 2026-07-01
source_url: https://help.ledgerly.example/integrations/quickbooks
---
# QuickBooks integration

Ledgerly connects to QuickBooks Online. QuickBooks Desktop is not supported. Xero and Sage are not supported at this time.

## What syncs

Customers and invoices sync in both directions every 15 minutes. Payments sync one way, from Ledgerly to QuickBooks. The first sync imports the last 12 months of data and handles up to 10,000 customers.

## Taxes and currencies

Tax codes are matched by name. Multi-currency invoices sync only if multi-currency is enabled in QuickBooks.

## Connection

QuickBooks authorisations expire, so you must re-authorise the connection every 100 days. Ledgerly emails the account owner 7 days before expiry. To disconnect, go to Settings, Integrations, Disconnect.
