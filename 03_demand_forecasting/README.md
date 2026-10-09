# 03 — Demand Forecasting with Prophet

This project implements the demand forecasting topic from the lecture.

The lecture says forecasting should help avoid:
- stockouts
- overstock

And should consider:
- trend
- seasonality
- cyclicity
- noise

## What the script does

1. Creates a daily product-demand dataset for a realistic demonstration.
2. Formats it for Prophet using:
   - `ds` = date
   - `y` = demand
3. Fits a Prophet model.
4. Forecasts the next 30 days.
5. Saves the forecast.
6. Generates a forecast chart.

## Run

```bash
python demand_forecasting.py
```

Outputs:
```text
outputs/demand_forecast.csv
outputs/demand_forecast.png
```
