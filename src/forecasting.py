import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def _build_daily_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    agg = {
        "daily_sales": "sum",
        "temperature": "mean",
        "precipitation": "mean",
        "consumer_confidence": "mean",
        "unemployment_rate": "mean",
        "competitor_count": "mean",
        "marketing_spend": "sum",
        "promotion_active": "mean",
        "avg_wait_time_min": "mean",
        "repeat_customer_pct": "mean",
        "labor_hours": "sum",
        "staff_on_shift": "sum",
    }
    available = [c for c in agg.keys() if c in df.columns]
    df_daily = df.groupby("date")[available].agg({c: agg[c] for c in available}).reset_index()

    df_daily = df_daily.sort_values("date")
    df_daily["day_of_week"] = df_daily["date"].dt.dayofweek
    df_daily["month"] = df_daily["date"].dt.month
    df_daily["is_weekend"] = df_daily["day_of_week"].isin([5, 6]).astype(int)
    return df_daily


def _default_future_features(history_daily: pd.DataFrame, future_dates: pd.DatetimeIndex) -> pd.DataFrame:
    hist = history_daily.sort_values("date").copy()
    feature_cols = [c for c in hist.columns if c not in {"date", "daily_sales"}]

    trailing = hist[feature_cols].tail(28).mean(numeric_only=True)
    future = pd.DataFrame({"date": future_dates})
    for c in feature_cols:
        future[c] = float(trailing.get(c, 0.0))

    future["day_of_week"] = future["date"].dt.dayofweek
    future["month"] = future["date"].dt.month
    future["is_weekend"] = future["day_of_week"].isin([5, 6]).astype(int)
    return future


def run_forecast(horizon_days: int = 30, test_days: int = 30) -> pd.DataFrame:

    df = pd.read_csv("data/model_features.csv")

    df_daily = _build_daily_model_frame(df)

    features = [
        "temperature",
        "precipitation",
        "consumer_confidence",
        "unemployment_rate",
        "competitor_count",
        "marketing_spend",
        "promotion_active",
        "avg_wait_time_min",
        "repeat_customer_pct",
        "labor_hours",
        "staff_on_shift",
        "is_weekend",
        "day_of_week",
        "month",
    ]
    features = [c for c in features if c in df_daily.columns]

    max_date = df_daily["date"].max()
    test_cutoff = max_date - pd.Timedelta(days=max(test_days, 1))
    train_mask = df_daily["date"] <= test_cutoff

    X_train = df_daily.loc[train_mask, features]
    y_train = df_daily.loc[train_mask, "daily_sales"]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    df_daily["prediction"] = model.predict(df_daily[features])
    df_daily["residual"] = df_daily["daily_sales"] - df_daily["prediction"]

    resid_std = float(df_daily.loc[train_mask, "residual"].std(ddof=0) or 0.0)
    if resid_std <= 0:
        resid_std = 1.0
    df_daily["anomaly"] = (df_daily["residual"].abs() >= 3.0 * resid_std).astype(int)
    df_daily["split"] = np.where(train_mask, "train", "test")

    if horizon_days and horizon_days > 0:
        future_dates = pd.date_range(max_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        df_future = _default_future_features(df_daily, future_dates)
        df_future["prediction"] = model.predict(df_future[features])
        df_future["daily_sales"] = np.nan
        df_future["residual"] = np.nan
        df_future["anomaly"] = 0
        df_future["split"] = "future"
        df_out = pd.concat([df_daily, df_future], ignore_index=True)
    else:
        df_out = df_daily

    df_out.to_csv("data/forecast_results.csv", index=False)
    print("Forecast complete")
    return df_out