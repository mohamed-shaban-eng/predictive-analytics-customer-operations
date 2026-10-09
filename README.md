# Predictive Analytics for Customer & Operations Management

A GitHub-ready Python project based on the lecture **"Chapter 3: Predictive Analytics for Customer and Operations Management"**.

The lecture covers:
1. Customer Lifetime Value (LTV) with BG/NBD + Gamma-Gamma
2. Customer Churn Prediction
3. Demand Forecasting with Prophet
4. Dynamic Pricing with Rule-Based logic + Reinforcement Learning concepts
5. Ethical considerations in pricing

The LTV section in the lecture defines LTV as expected total customer profit and decomposes it into:
**Predicted Future Transactions × Predicted Average Order Value × Average Gross Margin**.

## Project structure

```text
predictive-analytics-lecture-project/
├── 01_ltv_bg_nbd/
│   ├── README.md
│   └── ltv_analysis.py
├── 02_churn_prediction/
│   ├── README.md
│   └── churn_prediction.py
├── 03_demand_forecasting/
│   ├── README.md
│   └── demand_forecasting.py
├── 04_dynamic_pricing/
│   ├── README.md
│   └── dynamic_pricing.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Topics mapped to the lecture

### 01 — Customer Lifetime Value
Uses the CDNOW dataset available through `lifetimes`.

- Load CDNOW summary data with monetary value
- Train `BetaGeoFitter`
- Predict expected transactions for the next month
- Train `GammaGammaFitter`
- Estimate customer lifetime value for the next 12 weeks
- Rank the top 10 customers

### 02 — Churn Prediction
Uses the lecture's RFM + engagement idea.

Features:
- Recency
- Frequency
- Monetary
- Total clicks

The lecture labels a customer as churned when:
`days_since_last_purchase > 90`

The project then:
- splits data into train/test
- scales features
- trains `XGBClassifier`
- evaluates with classification metrics
- exports customers at risk of churn

### 03 — Demand Forecasting
Uses Prophet to forecast product demand while considering:
- trend
- seasonality
- cyclicity
- noise

The script creates a realistic demo demand dataset so the repository can run without an external CSV.

### 04 — Dynamic Pricing
Implements the two approaches shown in the lecture:
- Rule-based pricing
- Reinforcement learning (Q-learning demonstration)

Pricing reacts to:
- demand
- inventory
- competitor price

The project also includes practical fairness/ethics guardrails:
- bounded price changes
- no customer-personalized discriminatory pricing
- no competitor-collusion logic
- auditable pricing reasons

## Installation

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

Run each project:

```bash
python 01_ltv_bg_nbd/ltv_analysis.py
python 02_churn_prediction/churn_prediction.py
python 03_demand_forecasting/demand_forecasting.py
python 04_dynamic_pricing/dynamic_pricing.py
```

## Important dependency note

`lifetimes`, `prophet`, and `xgboost` are specialized scientific packages. If your current Python installation has binary/package compatibility problems, use a clean virtual environment with a Python version supported by the package versions you install.

## GitHub

Suggested repository name:

```text
predictive-analytics-customer-operations
```

Suggested description:

```text
Python projects covering LTV, BG/NBD, churn prediction, demand forecasting, and dynamic pricing based on a Predictive Analytics lecture.
```

## Source alignment

The implementation follows the terminology and workflow presented in the supplied lecture PDF, including:
- LTV
- CAC
- BG/NBD
- Gamma-Gamma
- churn/RFM/engagement
- Prophet
- dynamic pricing
- ethical pricing considerations
