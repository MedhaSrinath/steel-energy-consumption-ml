from __future__ import annotations

from pathlib import Path

import pandas as pd

from .model_training import load_model


def predict(model_path: str | Path, feature_frame: pd.DataFrame) -> pd.Series:
    """Generate predictions using a saved model."""
    model = load_model(model_path)
    predictions = model.predict(feature_frame)
    return pd.Series(predictions, index=feature_frame.index, name="prediction")
