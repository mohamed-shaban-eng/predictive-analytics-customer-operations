from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from xgboost import XGBClassifier


RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_customer_data(n_customers: int = 3000) -> pd.DataFrame:
    """
    Generate a realistic demo customer dataset using the same
    conceptual features used in the lecture.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    recency = rng.gamma(shape=2.5, scale=25, size=n_customers)
    recency = np.clip(recency, 1, 220)

    frequency = rng.poisson(lam=5, size=n_customers) + 1

    monetary = rng.lognormal(
        mean=np.log(80),
        sigma=0.65,
        size=n_customers,
    )

    total_clicks = rng.poisson(lam=18, size=n_customers)
    visits = rng.poisson(lam=10, size=n_customers)
    support_tickets = rng.poisson(lam=2.0, size=n_customers)

    # Construct a probability for demonstration.
    # Higher recency generally increases churn risk,
    # while stronger engagement reduces it.
    score = (
        -4.0
        + 0.040 * recency
        - 0.12 * frequency
        - 0.012 * total_clicks
        - 0.015 * visits
        + 0.10 * support_tickets
    )


    churn_probability = 1 / (1 + np.exp(-score))
    churn_label = rng.binomial(1, churn_probability)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:05d}" for i in range(1, n_customers + 1)],
            "recency": recency.round(2),
            "frequency": frequency,
            "monetary": monetary.round(2),
            "total_clicks": total_clicks,
            "visits": visits,
            "support_tickets": support_tickets,
            "is_churned": churn_label,
        }
    )

    return df


def main() -> None:
    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION")
    print("=" * 70)

    df = generate_customer_data()

    print("\nDataset preview:")
    print(df.head())

    print("\nChurn distribution:")
    print(df["is_churned"].value_counts().sort_index())

    # --------------------------------------------------------
    # Label exactly as shown in the lecture
    # --------------------------------------------------------
    # df["is_churned"] = df["recency"].apply(
    #     lambda x: 1 if x > 90 else 0
    # )

    # --------------------------------------------------------
    # Features shown in the lecture
    # --------------------------------------------------------
    feature_columns = [
        "recency",
        "frequency",
        "monetary",
        "total_clicks",
    ]

    X = df[feature_columns]
    y = df["is_churned"]

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # --------------------------------------------------------
    # Scale the features as shown in the lecture
    # --------------------------------------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------------
    # XGBoost classifier
    # --------------------------------------------------------
    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_scaled, y_train)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    print("\nClassification report:")
    print(classification_report(y_test, predictions, digits=4))

    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, probabilities)
        print(f"ROC-AUC: {auc:.4f}")

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------
    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Active", "Churned"],
    ).plot(ax=ax)
    ax.set_title("Churn Prediction - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------
    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns,
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    importance.plot(kind="barh", ax=ax)
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    # --------------------------------------------------------
    # Predict churn for all customers
    # --------------------------------------------------------
    all_scaled = scaler.transform(df[feature_columns])
    df["churn_probability"] = model.predict_proba(all_scaled)[:, 1]
    df["predicted_churn"] = model.predict(all_scaled)

    at_risk = (
        df.sort_values("churn_probability", ascending=False)
        .copy()
    )

    output_path = OUTPUT_DIR / "churn_predictions.csv"
    at_risk.to_csv(output_path, index=False)

    print("\nTop 20 customers by churn probability:")
    print(
        at_risk
        .head(20)
        .to_string(index=False)
    )

    print(f"\nSaved: {output_path}")
    print("Saved: confusion_matrix.png")
    print("Saved: feature_importance.png")


if __name__ == "__main__":
    main()
