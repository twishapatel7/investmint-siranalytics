from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AnalysisConfig:
    top_n: int = 6
    show_small_contrib_threshold: float = 0.03  # 3% of total abs contribution


def _fmt_money(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:,.0f}"


def _fmt_signed_money(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):,.0f}"


def _humanize_driver(name: str) -> str:
    return name.replace("_", " ").strip()


def build_written_analysis_for_date(
    rca_daily: pd.DataFrame, selected_date: pd.Timestamp, config: AnalysisConfig | None = None
) -> str:
    """
    Rule-based narrative from RCA contributions.
    No ML/LLM text generation; it just formats computed contributions into a readable summary.
    """
    config = config or AnalysisConfig()
    df = rca_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    row_df = df[df["date"] == pd.to_datetime(selected_date)]
    if row_df.empty:
        return "No RCA data available for the selected date."
    row = row_df.iloc[0]

    contrib_cols = [c for c in df.columns if c.endswith("__contribution")]
    if not contrib_cols:
        return "RCA contributions not found in the dataset."

    contrib = pd.Series({c.replace("__contribution", ""): float(row[c]) for c in contrib_cols})
    contrib = contrib.sort_values(key=lambda s: s.abs(), ascending=False)

    abs_total = float(contrib.abs().sum()) or 1.0
    top = contrib.head(config.top_n)

    # Filter out very small items (relative to total)
    top_filtered = top[top.abs() >= config.show_small_contrib_threshold * abs_total]
    if top_filtered.empty:
        top_filtered = top

    actual = float(row.get("daily_sales", float("nan")))
    baseline = float(row.get("baseline_daily_sales", float("nan")))
    delta = float(row.get("sales_delta_vs_baseline", float("nan")))

    lines: list[str] = []
    lines.append(f"Date: {pd.to_datetime(selected_date).date()}")
    lines.append(f"Actual sales: {_fmt_money(actual)}")
    lines.append(f"Baseline sales (trailing): {_fmt_money(baseline)}")
    lines.append(f"Delta vs baseline: {_fmt_signed_money(delta)}")
    lines.append("")

    # Headline: top positive/negative drivers
    pos = contrib[contrib > 0].head(3)
    neg = contrib[contrib < 0].head(3)

    if len(pos) > 0:
        lines.append("Main drivers increasing sales:")
        for k, v in pos.items():
            lines.append(f"- {_humanize_driver(k)}: {_fmt_signed_money(v)} contribution")
        lines.append("")

    if len(neg) > 0:
        lines.append("Main drivers decreasing sales:")
        for k, v in neg.items():
            lines.append(f"- {_humanize_driver(k)}: {_fmt_signed_money(v)} contribution")
        lines.append("")

    lines.append("What to pay attention to (largest contributors vs baseline):")
    for k, v in top_filtered.items():
        direction = "upside" if v >= 0 else "downside"
        lines.append(f"- {_humanize_driver(k)}: {_fmt_signed_money(v)} ({direction})")

    # If anomaly flag exists, add a simple callout
    anomaly = row.get("anomaly", 0)
    residual = row.get("residual", float("nan"))
    if pd.notna(anomaly) and int(anomaly) == 1:
        lines.append("")
        lines.append(f"Flag: anomaly day (model residual {_fmt_signed_money(float(residual))}).")

    return "\n".join(lines)


def build_summary_across_anomalies(rca_anoms: pd.DataFrame, top_n: int = 5) -> str:
    """
    Simple frequency-based summary: which drivers show up most often in anomaly explanations.
    This remains rule-based (counts) and doesn't use AI.
    """
    if rca_anoms is None or rca_anoms.empty:
        return "No anomalies detected in the current period."

    if "top_drivers" not in rca_anoms.columns:
        return "No anomaly explanations available."

    # count driver mentions
    counts: dict[str, int] = {}
    for s in rca_anoms["top_drivers"].fillna("").astype(str).tolist():
        parts = [p.strip() for p in s.split(",") if p.strip()]
        for p in parts:
            counts[p] = counts.get(p, 0) + 1

    if not counts:
        return "No anomaly explanations available."

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    lines = ["Drivers most frequently associated with anomaly days:"]
    for name, n in ranked:
        lines.append(f"- {_humanize_driver(name)}: mentioned {n} times")
    return "\n".join(lines)

