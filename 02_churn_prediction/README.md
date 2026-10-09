# 02 — Customer Churn Prediction

This project follows the churn section of the lecture.

## Churn definition

The lecture's labeling logic is:

```python
1 if days_since_last_purchase > 90 else 0
```

So the demo labels:
- `1` = churned
- `0` = active

## Features

The lecture emphasizes:

### RFM
- Recency
- Frequency
- Monetary

### Engagement
- Clicks
- Visits
- Support tickets

The main model uses the lecture's displayed features:
- recency
- frequency
- monetary
- total_clicks

The script also generates visits and support tickets for analysis.

## Model

- Train/test split
- StandardScaler
- XGBoost classifier
- Classification report
- Confusion matrix
- Feature importance

## Run

```bash
python churn_prediction.py
```

Outputs:
```text
outputs/churn_predictions.csv
outputs/confusion_matrix.png
outputs/feature_importance.png
```
