import pandas as pd


def load_data():

    stores = pd.read_csv("data/restaurant_stores.csv")
    ops = pd.read_csv("data/restaurant_daily_operations.csv")
    weather = pd.read_csv("data/restaurant_weather.csv")
    econ = pd.read_csv("data/restaurant_economics.csv")
    comp = pd.read_csv("data/restaurant_competition.csv")

    # convert date columns
    ops["date"] = pd.to_datetime(ops["date"])
    weather["date"] = pd.to_datetime(weather["date"])
    econ["date"] = pd.to_datetime(econ["date"])
    comp["date"] = pd.to_datetime(comp["date"])

    return stores, ops, weather, econ, comp


def build_dataset():

    stores, ops, weather, econ, comp = load_data()

    # -------------------
    # operations + stores
    # -------------------
    df = ops.merge(stores, on="store_id")

    # -------------------
    # weather (city level)
    # -------------------
    df = df.merge(
        weather,
        on=["city", "date"],
        how="left"
    )

    # -------------------
    # economic indicators
    # -------------------
    df = df.merge(
        econ,
        on=["city", "date"],
        how="left"
    )

    # -------------------
    # competition data
    # -------------------
    df = df.merge(
        comp,
        on=["store_id", "date"],
        how="left"
    )

    # -------------------
    # save dataset
    # -------------------
    df.to_csv("data/model_dataset.csv", index=False)

    print("\nDataset built successfully")
    print("Dataset shape:", df.shape)
    print(df.head())

    return df


if __name__ == "__main__":
    build_dataset()