import pandas as pd


def create_features():

    df = pd.read_csv("data/model_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])

    # -------------------
    # normalize core KPI / driver column names expected by downstream code
    # -------------------
    # KPIs
    if "daily_sales" not in df.columns and "revenue" in df.columns:
        df["daily_sales"] = df["revenue"]
    if "customer_count" not in df.columns and "orders" in df.columns:
        df["customer_count"] = df["orders"]

    # weather
    if "temperature" not in df.columns and "temperature_f" in df.columns:
        df["temperature"] = df["temperature_f"]
    if "precipitation" not in df.columns and "precipitation_in" in df.columns:
        df["precipitation"] = df["precipitation_in"]

    # economics
    if "consumer_confidence" not in df.columns and "consumer_sentiment" in df.columns:
        df["consumer_confidence"] = df["consumer_sentiment"]

    # competition
    if "competitor_count" not in df.columns:
        if "competitors_within_1mi" in df.columns:
            df["competitor_count"] = df["competitors_within_1mi"]
        elif "competitors_within_3mi" in df.columns:
            df["competitor_count"] = df["competitors_within_3mi"]
        else:
            df["competitor_count"] = 0

    # -------------------
    # time features
    # -------------------

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # -------------------
    # operational features
    # -------------------

    if "revenue" in df.columns and "orders" in df.columns:
        df["avg_ticket"] = df["revenue"] / (df["orders"] + 1)
    else:
        df["avg_ticket"] = 0

    df["revenue_per_labor_hour"] = (
        df["revenue"] / (df["labor_hours"] + 1)
    )

    df["marketing_efficiency"] = (
        df["revenue"] / (df["marketing_spend"] + 1)
    )

    # -------------------
    # competition pressure
    # automatically detect column names
    # -------------------

    comp_cols = [c for c in df.columns if "competitor" in c.lower()]

    if len(comp_cols) >= 2:

        c1 = comp_cols[0]
        c2 = comp_cols[1]

        df["competition_pressure"] = (
            df[c1] * 2 + df[c2]
        )

    else:

        df["competition_pressure"] = 0

    # -------------------
    # save dataset
    # -------------------

    # downstream scripts + dashboard expect this filename
    df.to_csv("data/model_features.csv", index=False)
    # keep the original artifact name for compatibility (same contents)
    df.to_csv("data/feature_dataset.csv", index=False)

    print("\nFeatures created successfully")
    print("Feature dataset shape:", df.shape)

    return df