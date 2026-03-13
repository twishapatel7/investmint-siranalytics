from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_daily_frame() -> pd.DataFrame:
    df = pd.read_csv("data/model_features.csv")
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


def _fit_explainable_model(df_daily: pd.DataFrame, features: list[str], test_days: int = 30) -> Pipeline:
    max_date = df_daily["date"].max()
    cutoff = max_date - pd.Timedelta(days=max(test_days, 1))
    train = df_daily[df_daily["date"] <= cutoff]

    X_train = train[features]
    y_train = train["daily_sales"]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def build_root_cause_artifacts(test_days: int = 30, baseline_days: int = 28) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produces two artifacts:
      - data/rca_daily_contributions.csv: per-day driver deltas vs baseline and their attributed contribution to sales delta
      - data/rca_anomalies.csv: subset of anomaly days with top drivers
    """
    df_daily = _build_daily_frame()

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

    model = _fit_explainable_model(df_daily, features=features, test_days=test_days)
    scaler: StandardScaler = model.named_steps["scaler"]
    ridge: Ridge = model.named_steps["ridge"]

    # baseline: trailing window mean (in raw feature space)
    baseline_raw = df_daily[features].tail(baseline_days).mean(numeric_only=True)
    baseline_raw = baseline_raw.reindex(features).fillna(0.0)

    # compute per-day deltas in standardized space so contributions are in sales units
    X = df_daily[features].fillna(0.0)
    X_scaled = scaler.transform(X)
    baseline_scaled = scaler.transform(pd.DataFrame([baseline_raw.values], columns=features))[0]
    delta_scaled = X_scaled - baseline_scaled

    coefs = ridge.coef_.reshape(1, -1)
    contrib = delta_scaled * coefs  # (n_days, n_features)

    out = df_daily[["date", "daily_sales"]].copy()
    out["baseline_daily_sales"] = float(df_daily["daily_sales"].tail(baseline_days).mean())
    out["sales_delta_vs_baseline"] = out["daily_sales"] - out["baseline_daily_sales"]

    for i, f in enumerate(features):
        out[f"{f}__delta_vs_baseline"] = X[f] - float(baseline_raw[f])
        out[f"{f}__contribution"] = contrib[:, i]

    # attach anomaly flags from the forecast artifact if present
    try:
        fc = pd.read_csv("data/forecast_results.csv")
        fc["date"] = pd.to_datetime(fc["date"])
        fc = fc[fc["split"].isin(["train", "test"])][["date", "anomaly", "residual"]]
        out = out.merge(fc, on="date", how="left")
    except Exception:
        out["anomaly"] = 0
        out["residual"] = np.nan

    out.to_csv("data/rca_daily_contributions.csv", index=False)

    # build anomaly explanations (top 5 contributors by magnitude)
    anomalies = out[out["anomaly"] == 1].copy()
    if len(anomalies) > 0:
        contrib_cols = [f"{f}__contribution" for f in features]

        def _top_drivers(row: pd.Series) -> str:
            s = row[contrib_cols].astype(float)
            top = s.abs().sort_values(ascending=False).head(5).index.tolist()
            named = [c.replace("__contribution", "") for c in top]
            return ", ".join(named)

        anomalies["top_drivers"] = anomalies.apply(_top_drivers, axis=1)
    else:
        anomalies["top_drivers"] = ""

    anomalies.to_csv("data/rca_anomalies.csv", index=False)
    return out, anomalies

