# Abstention gate calibration

Best vector cosine per question (44 answerable, 14 unanswerable). A similarity gate refuses a question without calling the model when the best cosine is below the threshold.

| threshold | answerable wrongly blocked (of 44) | unanswerable caught (of 14) |
|---|---|---|
| 0.5 | 0 | 0 |
| 0.525 | 0 | 0 |
| 0.55 | 0 | 0 |
| 0.575 | 0 | 2 |
| 0.6 | 1 | 4 |
| 0.625 | 1 | 5 |
| 0.65 | 3 | 7 |
| 0.675 | 6 | 7 |
| 0.7 | 12 | 8 |
| 0.725 | 16 | 11 |
| 0.75 | 22 | 13 |

Lowest-scoring answerable questions (these decide how high the gate can safely be):

- 0.587  q41: A client's bank wants an extra verification step before taking the money. What happens?
- 0.645  q39: If someone stole the hard drives, could they read my customers' information?
- 0.646  q47: Can my whole team log in through the company's central login system?
- 0.659  q48: We want to try before buying. How long is it and do we need to give payment details?
- 0.665  q45: If your platform is down for hours, do I get compensated?
- 0.666  q07: How long does a refund take to reach my card?

Unanswerable questions:

- 0.563  q51: Can I pay my subscription with cryptocurrency?
- 0.565  q49: Do you have an iOS or Android mobile app?
- 0.579  q33: What is the CEO's email address?
- 0.59  q32: Do you support payouts in Brazilian real?
- 0.602  q57: How do I export all my invoices to a CSV file?
- 0.628  q60: Which countries are supported for payouts?
- 0.643  q52: What does the Enterprise plan cost?
- 0.685  q59: What is the maximum file size for invoice attachments?
- 0.707  q53: Is the Ledgerly interface available in Brazilian Portuguese?
- 0.707  q56: What is the phone number for Scale plan support?
- 0.717  q58: Does Ledgerly have an affiliate or referral program?
- 0.737  q54: Who are Ledgerly's investors?
- 0.747  q50: How many employees does Ledgerly have?
- 0.783  q55: Does Ledgerly integrate with Slack?
