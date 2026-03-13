import pandas as pd
from sklearn.linear_model import LinearRegression


def run_regression():

    df = pd.read_csv("data/model_features.csv")

    features = [
        "temperature",
        "precipitation",
        "unemployment_rate",
        "consumer_confidence",
        "competitor_count",
        "is_weekend"
    ]

    X = df[features]
    y = df["daily_sales"]

    model = LinearRegression()
    model.fit(X, y)

    results = pd.DataFrame({
        "feature": features,
        "impact": model.coef_
    })

    results = results.sort_values("impact", ascending=False)

    print("\nSales Drivers:")
    print(results)

    return results