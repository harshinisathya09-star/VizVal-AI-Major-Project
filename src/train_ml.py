from __future__ import annotations

import json
import platform
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - handled in UI/CLI.
    XGBRegressor = None

from .data_loader import load_dataset
from .feature_engineering import add_features
from .preprocessing import build_preprocessor, clean_property_data, model_matrix, validate_dataset
from .utils import DATA_PROCESSED_DIR, MODELS_DIR, RANDOM_STATE, ensure_directories


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100),
    }


def train_ml_models(csv_path: str | None = None) -> dict[str, object]:
    ensure_directories()
    raw_df, source_message = load_dataset(csv_path)
    quality = validate_dataset(raw_df)
    df = add_features(clean_property_data(raw_df), include_price_features=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED_DIR / "training_data.csv", index=False)

    X, y = model_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=220,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=260,
            learning_rate=0.055,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
        )

    results = []
    trained = {}
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        metrics = regression_metrics(y_test, preds)
        results.append({"Model": name, **metrics})
        trained[name] = pipeline

    metrics_df = pd.DataFrame(results).sort_values("RMSE")
    best_name = str(metrics_df.iloc[0]["Model"])
    residuals = np.abs(y_test - trained[best_name].predict(X_test))
    interval_error = float(np.quantile(residuals, 0.90))

    file_map = {
        "Linear Regression": "linear_regression_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "XGBoost": "xgboost_model.pkl",
    }
    for name, model in trained.items():
        joblib.dump(model, MODELS_DIR / file_map[name])
    joblib.dump(trained[best_name].named_steps["preprocessor"], MODELS_DIR / "preprocessor.pkl")

    metadata = {
        "source_message": source_message,
        "best_ml_model": best_name,
        "interval_error_90": interval_error,
        "quality": {k: v for k, v in quality.items() if k not in {"numeric_summary"}},
        "metrics": metrics_df.to_dict(orient="records"),
        "model_files": file_map,
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    model_metadata = {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "xgboost_version": getattr(__import__("xgboost"), "__version__", None)
        if XGBRegressor is not None
        else None,
        "training_timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(model_metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


if __name__ == "__main__":
    summary = train_ml_models()
    print(json.dumps(summary, indent=2))
