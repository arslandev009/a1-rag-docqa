---
title: Troubleshooting Failed Payments
doc_type: troubleshooting
version: "2026-07"
status: current
audience: public
effective_date: 2026-07-01
source_url: https://help.ledgerly.example/payments/failed
---
# Troubleshooting failed payments

## Common error codes

- **card_declined:** the bank refused the card. Ask the customer to use another card.
- **insufficient_funds:** the card has too little balance. The customer should retry or use another card.
- **expired_card:** the card is past its expiry date. The customer must update the card.
- **authentication_required:** the bank needs 3-D Secure verification. The customer must complete the authentication step in the payment link.
- **processing_error:** a temporary problem. Ledgerly retries automatically up to 3 times at 24-hour intervals.

## Automatic retries and reminders

After a failed payment Ledgerly retries on day 1, day 3 and day 7. If all retries fail the invoice is marked overdue. Reminder emails go out on day 0, day 3 and day 7. Growth and Scale customers can edit the reminder text.

## Payouts and disputes

Standard payouts arrive in 2 business days. Instant payouts carry a 1% fee. A chargeback costs $15 per dispute.
