from __future__ import annotations

import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from .feature_engineering import add_features
from .utils import MODEL_FEATURES, MODELS_DIR


MODEL_FILES = {
    "Linear Regression": "linear_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl",
}


def load_model_metadata() -> dict:
    metadata_path = MODELS_DIR / "model_metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def current_runtime_metadata() -> dict:
    try:
        import xgboost

        xgboost_version = xgboost.__version__
    except ImportError:
        xgboost_version = None
    return {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "xgboost_version": xgboost_version,
    }


def artifact_environment_matches() -> bool:
    saved = load_model_metadata()
    if not saved:
        return False
    current = current_runtime_metadata()
    keys = ["python_version", "sklearn_version", "pandas_version", "numpy_version", "xgboost_version"]
    return all(saved.get(key) == current.get(key) for key in keys)


def models_available() -> bool:
    return (
        (MODELS_DIR / "metrics.json").exists()
        and artifact_environment_matches()
        and all(
        (MODELS_DIR / filename).exists() for filename in MODEL_FILES.values()
        )
    )


def load_metrics() -> dict:
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return {"metrics": [], "best_ml_model": None, "interval_error_90": 0}


def load_ml_models() -> dict[str, object]:
    if not artifact_environment_matches():
        raise RuntimeError(
            "Saved model artifacts are missing metadata or were trained with a different runtime. "
            "Run python3 -m src.train_ml to regenerate compatible models."
        )
    loaded = {}
    for name, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        if path.exists():
            loaded[name] = joblib.load(path)
    return loaded


def load_deep_learning_model():
    model_path = MODELS_DIR / "neural_network.keras"
    preprocessor_path = MODELS_DIR / "dl_preprocessor.pkl"
    metrics_path = MODELS_DIR / "dl_metrics.json"
    if not (model_path.exists() and preprocessor_path.exists() and metrics_path.exists()):
        return None
    try:
        import tensorflow as tf
    except ImportError:
        return None
    return {
        "model": tf.keras.models.load_model(model_path),
        "preprocessor": joblib.load(preprocessor_path),
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
    }


def prepare_input(record: dict) -> pd.DataFrame:
    df = pd.DataFrame([record])
    df = add_features(df, include_price_features=False)
    return df[MODEL_FEATURES]


def prediction_weights(ml_rmse: float | None, dl_rmse: float | None) -> tuple[float, float]:
    if not ml_rmse and not dl_rmse:
        return 1.0, 0.0
    if not dl_rmse or dl_rmse <= 0:
        return 1.0, 0.0
    if not ml_rmse or ml_rmse <= 0:
        return 0.0, 1.0
    ml_inv = 1 / ml_rmse
    dl_inv = 1 / dl_rmse
    total = ml_inv + dl_inv
    return ml_inv / total, dl_inv / total


def predict_property(record: dict) -> dict[str, float | str | None]:
    metrics = load_metrics()
    models = load_ml_models()
    if not models:
        raise FileNotFoundError("No trained ML models found. Run python3 -m src.train_ml first.")

    best_name = metrics.get("best_ml_model") or next(iter(models))
    if best_name not in models:
        best_name = next(iter(models))
    X = prepare_input(record)
    ml_prediction = float(models[best_name].predict(X)[0])

    dl_bundle = load_deep_learning_model()
    dl_prediction = None
    dl_rmse = None
    if dl_bundle:
        transformed = dl_bundle["preprocessor"].transform(X)
        scale = float(dl_bundle["metrics"].get("target_scale", 1))
        dl_prediction = float(dl_bundle["model"].predict(transformed, verbose=0).ravel()[0] * scale)
        dl_rmse = float(dl_bundle["metrics"].get("RMSE", 0))

    ml_rmse = None
    for row in metrics.get("metrics", []):
        if row.get("Model") == best_name:
            ml_rmse = float(row.get("RMSE", 0))

    ml_weight, dl_weight = prediction_weights(ml_rmse, dl_rmse)
    if dl_prediction is None:
        final_prediction = ml_prediction
    else:
        final_prediction = ml_prediction * ml_weight + dl_prediction * dl_weight

    interval_error = float(metrics.get("interval_error_90", 0) or 0)
    return {
        "best_ml_model": best_name,
        "ml_prediction": ml_prediction,
        "dl_prediction": dl_prediction,
        "final_prediction": float(final_prediction),
        "price_per_sqft": float(final_prediction / max(float(record["area_sqft"]), 1)),
        "range_low": float(max(0, final_prediction - interval_error)),
        "range_high": float(final_prediction + interval_error),
        "ml_weight": ml_weight,
        "dl_weight": dl_weight,
    }
