from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_demo_demand_data(days: int = 365) -> pd.DataFrame:
    """
    Create daily demand with trend + weekly seasonality + yearly-like
    smooth seasonality + noise.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    dates = pd.date_range(
        start="2025-01-01",
        periods=days,
        freq="D",
    )

    t = np.arange(days)

    trend = 0.08 * t
    weekly = 10 * np.sin(2 * np.pi * t / 7)
    yearly = 16 * np.sin(2 * np.pi * t / 365)
    noise = rng.normal(0, 5, size=days)

    demand = (
        120
        + trend
        + weekly
        + yearly
        + noise
    )

    demand = np.maximum(demand, 5).round()

    return pd.DataFrame(
        {
            "date": dates,
            "sales": demand,
        }
    )


def main() -> None:
    print("=" * 70)
    print("DEMAND FORECASTING WITH PROPHET")
    print("=" * 70)

    df = create_demo_demand_data()

    print("\nRaw data:")
    print(df.head())

    # Prophet requires:
    # ds = date
    # y  = target value
    prophet_df = df.rename(
        columns={
            "date": "ds",
            "sales": "y",
        }
    )
    print("#"*10)
    print(prophet_df)
    print("#"*10)
    

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
    )

    model.fit(prophet_df)

    # Forecast 30 future days as an example
    future = model.make_future_dataframe(
        periods=30,
        freq="D",
    )
    print(future)
    print("#"*10)

    forecast = model.predict(future)

    print(forecast)
    print("#"*10)

    # Save the useful columns
    output = forecast[
        [
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper",
        ]
    ].copy()

    print(output)
    print("#"*10)

    output_path = OUTPUT_DIR / "demand_forecast.csv"
    output.to_csv(output_path, index=False)

    # --------------------------------------------------------
    # Forecast visualization
    # --------------------------------------------------------
    fig = model.plot(forecast)
    fig.set_size_inches(12, 6)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "demand_forecast.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Components
    components_fig = model.plot_components(forecast)
    components_fig.savefig(
        OUTPUT_DIR / "forecast_components.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(components_fig)

    # --------------------------------------------------------
    # Print next 30 days
    # --------------------------------------------------------
    future_only = output.tail(30)

    print("\nNext 30 days forecast:")
    print(future_only.to_string(index=False))

    print(f"\nSaved: {output_path}")
    print("Saved: demand_forecast.png")
    print("Saved: forecast_components.png")


if __name__ == "__main__":
    main()
