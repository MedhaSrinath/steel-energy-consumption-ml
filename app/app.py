from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import build_features
from src.model_training import save_model, train_model


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "steel_industry_data.csv"
MODEL_PATH = ROOT_DIR / "models" / "model.pkl"
DEFAULT_TARGET = "Usage_kWh"


def _load_dataset(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if DATA_PATH.exists():
        return load_data(DATA_PATH)
    return pd.DataFrame()


def main() -> None:
    st.set_page_config(page_title="Steel Energy Consumption ML", layout="wide")
    st.title("Steel Energy Consumption ML")
    st.write("Upload the dataset or use the local CSV to train a baseline model.")

    uploaded_file = st.file_uploader("Upload steel industry CSV", type=["csv"])
    target_column = st.text_input("Target column", value=DEFAULT_TARGET)

    frame = _load_dataset(uploaded_file)
    if frame.empty:
        st.info("No dataset found yet. Add data/steel_industry_data.csv or upload a CSV to begin.")
        return

    st.subheader("Preview")
    st.dataframe(frame.head())

    cleaned_frame = preprocess_data(frame)
    if target_column not in cleaned_frame.columns:
        st.warning(
            f"Target column '{target_column}' was not found after preprocessing. "
            "Update the target column and try again."
        )
        return

    features, target = build_features(cleaned_frame, target_column=target_column)
    if target is None:
        st.warning("Unable to build a target series from the dataset.")
        return

    model, metrics = train_model(features, target)
    save_model(model, MODEL_PATH)

    metric_left, metric_middle, metric_right = st.columns(3)
    metric_left.metric("MAE", f"{metrics['mae']:.4f}")
    metric_middle.metric("RMSE", f"{metrics['rmse']:.4f}")
    metric_right.metric("R²", f"{metrics['r2']:.4f}")

    st.success(f"Model saved to {MODEL_PATH}")
    st.subheader("Feature matrix")
    st.dataframe(features.head())


if __name__ == "__main__":
    main()
