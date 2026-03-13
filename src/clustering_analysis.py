import pandas as pd
from sklearn.cluster import KMeans


def run_clustering():

    # clustering needs the normalized KPI/driver columns
    # (daily_sales, customer_count, temperature) which are created in model_features.csv
    df = pd.read_csv("data/model_features.csv")

    store_metrics = df.groupby("store_id").agg({
        "daily_sales": "mean",
        "customer_count": "mean",
        "temperature": "mean"
    }).reset_index()

    X = store_metrics[[
        "daily_sales",
        "customer_count",
        "temperature"
    ]]

    kmeans = KMeans(n_clusters=3, random_state=42)

    store_metrics["cluster"] = kmeans.fit_predict(X)

    store_metrics.to_csv("data/store_clusters.csv", index=False)

    print("Clustering complete")

    return store_metrics