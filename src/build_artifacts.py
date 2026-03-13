"""
Build all CSV artifacts required by the Streamlit dashboard.

Usage:
  python3 -m src.build_artifacts
"""

from __future__ import annotations


def main() -> None:
    from src.data_pipeline import build_dataset
    from src.feature_engineering import create_features
    from src.clustering_analysis import run_clustering
    from src.forecasting import run_forecast
    from src.root_cause import build_root_cause_artifacts

    build_dataset()
    create_features()
    run_clustering()
    run_forecast()
    build_root_cause_artifacts()


if __name__ == "__main__":
    main()

