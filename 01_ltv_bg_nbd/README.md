# 01 — Customer Lifetime Value with BG/NBD + Gamma-Gamma

This project implements the LTV workflow shown in the lecture.

## Models

### BG/NBD
Predicts future transaction frequency from:
- frequency
- recency
- T

The lecture explains:
- Frequency = number of repeat operations/purchases
- Recency = time since purchasing activity in the observation period
- T = observation time

### Gamma-Gamma
Models average monetary value per transaction.

### LTV

The lecture presents the conceptual decomposition:

```text
LTV =
Predicted Future Transactions
× Predicted Average Order Value
× Average Gross Margin
```

The script additionally prints the top 10 customers by predicted 12-week LTV.

## Run

```bash
python ltv_analysis.py
```

Outputs are saved to:

```text
outputs/ltv_top10.csv
outputs/ltv_all_customers.csv
```
