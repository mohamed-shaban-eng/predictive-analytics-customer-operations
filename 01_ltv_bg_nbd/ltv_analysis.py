from pathlib import Path

import pandas as pd
from lifetimes.datasets import load_cdnow_summary_data_with_monetary_value
# نموذج التكرار والانقطاع
from lifetimes import BetaGeoFitter  # Expected future transactions in next T periods., probability of being alive, and dropout rate for each customer
# نموذج القيمة المالية
from lifetimes import  GammaGammaFitter # what AOV will be for a customer with a given frequency and monetary value



OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Load the CDNOW dataset used in the lecture
# ------------------------------------------------------------
data = load_cdnow_summary_data_with_monetary_value()

print("=" * 70)
print("CUSTOMER LIFETIME VALUE - BG/NBD + GAMMA-GAMMA")
print("=" * 70)

print("\nFirst rows:")
print(data.head())

print("\nColumns:")
print(data.columns.tolist())

# ------------------------------------------------------------
# 2. Basic cleaning
# ------------------------------------------------------------
required_columns = ["frequency", "recency", "T", "monetary_value"]

missing = [col for col in required_columns if col not in data.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

data = data.copy()

# Gamma-Gamma requires positive monetary values for customers
# with repeat purchases. Keep only valid rows.
model_data = data[
    (data["frequency"] > 0)
    & (data["monetary_value"] > 0)
].copy()

print(f"\nCustomers in original dataset: {len(data):,}")
print(f"Customers used for Gamma-Gamma/LTV: {len(model_data):,}")

# ------------------------------------------------------------
# 3. Train BG/NBD model
# ------------------------------------------------------------
print("\nTraining BG/NBD model...")

bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(
    model_data["frequency"],
    model_data["recency"],
    model_data["T"],
)

# ------------------------------------------------------------
# 4. Predict expected transactions for the next month
# ------------------------------------------------------------
model_data["pred_txn_next_4_weeks"] = bgf.conditional_expected_number_of_purchases_up_to_time(
    28,  # 4 weeks
    model_data["frequency"],
    model_data["recency"],
    model_data["T"],
)

print("\nExpected transactions in the next 4 weeks:")
print(
    model_data[
        ["pred_txn_next_4_weeks"]
    ].describe()
)

# ------------------------------------------------------------
# 5. Train Gamma-Gamma model
# ------------------------------------------------------------
print("\nTraining Gamma-Gamma model...")

ggf = GammaGammaFitter(penalizer_coef=0.0)
ggf.fit(
    model_data["frequency"],
    model_data["monetary_value"],
)

# ------------------------------------------------------------
# 6. Estimate customer lifetime value for next 12 weeks
# ------------------------------------------------------------
print("\nEstimating 12-week customer lifetime value...")

model_data["pred_ltv_12_weeks"] = ggf.customer_lifetime_value(
    bgf,
    model_data["frequency"],
    model_data["recency"],
    model_data["T"],
    model_data["monetary_value"],
    time=12,
    freq="W",
    discount_rate=0.01,
)


# ------------------------------------------------------------
# 6.1 Estimate profit LTV for next 12 weeks
# ------------------------------------------------------------

# 1. تحديد نسبة هامش الربح (مثلاً 30%) علي كل المنتجات طبعاا افتراض ده
gross_margin_rate = 0.50

# 2. إنشاء العمود الجديد بضرب توقعات المبيعات في هامش الربح
model_data["profit_ltv_12_weeks"] = model_data["pred_ltv_12_weeks"] * gross_margin_rate

# 3. عرض النتيجة للتأكد من العمودين معاً
print("\nPredicted LTV and Profit LTV for next 12 weeks:")
print(model_data[["pred_ltv_12_weeks", "profit_ltv_12_weeks"]].head())


# ------------------------------------------------------------
# 7. Rank top 10 customers
# ------------------------------------------------------------
top10 = (
    model_data
    .nlargest(10, "pred_ltv_12_weeks")
    .reset_index()
)

print("\nTop 10 customers by predicted 12-week LTV:")
print(top10.to_string(index=False))

# ------------------------------------------------------------
# 8. Save results
# ------------------------------------------------------------
all_results_path = OUTPUT_DIR / "ltv_all_customers.csv"
top10_path = OUTPUT_DIR / "ltv_top10.csv"

model_data.reset_index().to_csv(all_results_path, index=False)
top10.to_csv(top10_path, index=False)

print(f"\nSaved: {all_results_path}")
print(f"Saved: {top10_path}")

print("\nDone.")
