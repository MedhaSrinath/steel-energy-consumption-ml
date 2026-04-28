from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int = 42,
) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """Train a baseline regression model and return evaluation metrics."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions, squared=False)),
        "r2": float(r2_score(y_test, predictions)),
    }
    return model, metrics


def save_model(model: RandomForestRegressor, model_path: str | Path) -> Path:
    """Persist a trained model to disk."""
    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def load_model(model_path: str | Path) -> RandomForestRegressor:
    """Load a persisted model from disk."""
    return joblib.load(Path(model_path))
