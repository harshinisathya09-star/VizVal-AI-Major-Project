from __future__ import annotations

import numpy as np
import pandas as pd

from .prediction import load_ml_models, load_metrics, prepare_input
from .utils import MODEL_FEATURES


def explain_prediction(record: dict, top_n: int = 10) -> pd.DataFrame:
    """Return model-influence estimates for display; SHAP is used when available."""
    metrics = load_metrics()
    models = load_ml_models()
    if not models:
        return pd.DataFrame(columns=["feature", "contribution"])
    best_name = metrics.get("best_ml_model") or next(iter(models))
    if best_name not in models:
        best_name = next(iter(models))
    model = models[best_name]
    X = prepare_input(record)

    try:
        import shap

        transformed = model.named_steps["preprocessor"].transform(X)
        estimator = model.named_steps["model"]
        explainer = shap.Explainer(estimator)
        values = explainer(transformed)
        names = model.named_steps["preprocessor"].get_feature_names_out()
        contributions = np.asarray(values.values[0], dtype=float)
        df = pd.DataFrame({"feature": names, "contribution": contributions})
    except Exception:
        estimator = model.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            names = model.named_steps["preprocessor"].get_feature_names_out()
            df = pd.DataFrame({"feature": names, "contribution": estimator.feature_importances_})
        elif hasattr(estimator, "coef_"):
            names = model.named_steps["preprocessor"].get_feature_names_out()
            df = pd.DataFrame({"feature": names, "contribution": np.ravel(estimator.coef_)})
        else:
            df = pd.DataFrame({"feature": MODEL_FEATURES, "contribution": np.zeros(len(MODEL_FEATURES))})

    df["feature"] = df["feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
    df["abs_contribution"] = df["contribution"].abs()
    return df.sort_values("abs_contribution", ascending=False).head(top_n)
