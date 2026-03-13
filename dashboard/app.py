import sys
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import plotly.graph_objects as go

# Ensure repo root is on sys.path so `import src...` works even if Streamlit's CWD differs.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reporting import build_written_analysis_for_date, build_summary_across_anomalies, AnalysisConfig

st.set_page_config(layout="wide")

st.title("Restaurant Performance Analytics")

tab1, tab2, tab3, tab4 = st.tabs([
    "Performance",
    "Drivers",
    "Forecast",
    "Root Cause",
])


DATA_DIR = Path("data")


def _ensure_artifacts() -> None:
    required = [
        DATA_DIR / "model_features.csv",
        DATA_DIR / "store_clusters.csv",
        DATA_DIR / "forecast_results.csv",
        DATA_DIR / "rca_daily_contributions.csv",
        DATA_DIR / "rca_anomalies.csv",
    ]
    if all(p.exists() for p in required):
        return

    with st.status("Building required artifacts…", expanded=True):
        from src.build_artifacts import main as build_main

        build_main()
        st.write("Artifacts built.")


_ensure_artifacts()

df = pd.read_csv(DATA_DIR / "model_features.csv")
clusters = pd.read_csv(DATA_DIR / "store_clusters.csv")
forecast = pd.read_csv(DATA_DIR / "forecast_results.csv")
rca_daily = pd.read_csv(DATA_DIR / "rca_daily_contributions.csv")
rca_anoms = pd.read_csv(DATA_DIR / "rca_anomalies.csv")


# -------------------------
# TAB 1: PERFORMANCE
# -------------------------

with tab1:

    st.subheader("Sales Over Time")

    sales = df.groupby("date")["daily_sales"].sum().reset_index()

    fig = px.line(
        sales,
        x="date",
        y="daily_sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    dim = "region" if "region" in df.columns else ("store_type" if "store_type" in df.columns else ("city" if "city" in df.columns else None))
    if dim is not None:
        st.subheader(f"Sales by {dim.replace('_', ' ').title()}")
        dim_sales = df.groupby(dim)["daily_sales"].sum().reset_index()
        fig = px.pie(dim_sales, names=dim, values="daily_sales")
        st.plotly_chart(fig)
    else:
        st.info("No dimension column (region/store_type/city) found for a breakdown chart.")


# -------------------------
# TAB 2: DRIVERS
# -------------------------

with tab2:

    st.subheader("Store Clusters")

    fig = px.scatter(
        clusters,
        x="customer_count",
        y="daily_sales",
        color="cluster"
    )

    st.plotly_chart(fig)


# -------------------------
# TAB 3: FORECAST
# -------------------------

with tab3:

    st.subheader("Prediction vs Actual")

    fig = px.line(
        forecast,
        x="date",
        y=["daily_sales", "prediction"]
    )

    st.plotly_chart(fig, use_container_width=True)

    if "anomaly" in forecast.columns:
        anoms = forecast[(forecast["anomaly"] == 1) & (forecast.get("split", "test").isin(["train", "test"]))]
        st.caption(f"Anomalies detected (|residual| >= 3σ): {len(anoms)} days")


# -------------------------
# TAB 4: ROOT CAUSE
# -------------------------

with tab4:

    st.subheader("Automatic Root Cause Detection (variance attribution)")

    rca_daily["date"] = pd.to_datetime(rca_daily["date"])
    rca_anoms["date"] = pd.to_datetime(rca_anoms["date"])

    left, right = st.columns([2, 1])

    with right:
        anomaly_dates = rca_anoms.sort_values("date", ascending=False)["date"].dt.date.astype(str).tolist()
        if anomaly_dates:
            selected = st.selectbox("Anomaly date", options=anomaly_dates)
        else:
            recent_dates = rca_daily.sort_values("date", ascending=False)["date"].dt.date.astype(str).head(60).tolist()
            selected = st.selectbox("Date", options=recent_dates)
        selected_date = pd.to_datetime(selected)

    row = rca_daily[rca_daily["date"] == selected_date]
    if row.empty:
        st.warning("No RCA data for the selected date.")
    else:
        row = row.iloc[0]

        contrib_cols = [c for c in rca_daily.columns if c.endswith("__contribution")]
        contrib = (
            pd.Series({c.replace("__contribution", ""): float(row[c]) for c in contrib_cols})
            .sort_values(key=lambda s: s.abs(), ascending=False)
        )
        top = contrib.head(8)

        with left:
            m1, m2, m3 = st.columns(3)
            m1.metric("Actual sales", f"{row['daily_sales']:,.0f}")
            m2.metric("Baseline sales", f"{row['baseline_daily_sales']:,.0f}")
            m3.metric("Delta vs baseline", f"{row['sales_delta_vs_baseline']:,.0f}")

            wf = go.Figure(
                go.Waterfall(
                    orientation="v",
                    measure=["relative"] * len(top) + ["total"],
                    x=list(top.index) + ["sum(top)"],
                    y=list(top.values) + [float(top.sum())],
                )
            )
            wf.update_layout(
                title="Top driver contributions (vs trailing baseline)",
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(wf, use_container_width=True)

        st.subheader("Written analysis (simple, rule-based)")
        analysis_text = build_written_analysis_for_date(
            rca_daily=rca_daily,
            selected_date=selected_date,
            config=AnalysisConfig(top_n=8),
        )
        st.text(analysis_text)
        st.download_button(
            "Download analysis (.txt)",
            data=analysis_text.encode("utf-8"),
            file_name=f"analysis_{selected_date.date()}.txt",
            mime="text/plain",
        )

        st.caption(build_summary_across_anomalies(rca_anoms, top_n=6))

        st.subheader("Anomaly list")
        cols = [c for c in ["date", "daily_sales", "residual", "top_drivers"] if c in rca_anoms.columns]
        st.dataframe(rca_anoms.sort_values("date", ascending=False)[cols], use_container_width=True)