from __future__ import annotations

import json
import os

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .data_loader import load_dataset
from .feature_engineering import add_features
from .preprocessing import build_preprocessor, clean_property_data, model_matrix
from .utils import MODELS_DIR, RANDOM_STATE, ensure_directories

_mpl_cache = MODELS_DIR.parent / ".cache" / "matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))


def train_deep_learning_model(csv_path: str | None = None) -> dict[str, float | str]:
    try:
        import tensorflow as tf
        from keras.layers import Dense, Dropout, Input
        from keras.models import Sequential
        from keras.optimizers import Adam
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow/Keras is required for the deep learning workflow. "
            "Install requirements.txt and run this command again."
        ) from exc

    ensure_directories()
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    raw_df, _ = load_dataset(csv_path)
    df = add_features(clean_property_data(raw_df), include_price_features=True)
    if len(df) > 700:
        df = df.sample(700, random_state=RANDOM_STATE).reset_index(drop=True)
    X, y = model_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    y_scale = float(y_train.max())
    y_train_s = y_train / y_scale

    model = Sequential(
        [
            Input(shape=(X_train_t.shape[1],)),
            Dense(128, activation="relu"),
            Dropout(0.20),
            Dense(64, activation="relu"),
            Dropout(0.15),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    # A compact explicit training loop keeps classroom demos fast while still
    # training the Keras network on preprocessed property examples.
    batch_size = 128
    for epoch in range(12):
        order = np.random.default_rng(RANDOM_STATE + epoch).permutation(len(X_train_t))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            model.train_on_batch(X_train_t[idx], y_train_s.iloc[idx].to_numpy())

    preds = model.predict(X_test_t, verbose=0).ravel() * y_scale
    metrics = {
        "Model": "Neural Network",
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
        "target_scale": y_scale,
        "location_embedding_note": (
            "One-hot encoding was used. Embeddings are skipped unless a larger real dataset "
            "has enough repeated examples for each location."
        ),
    }
    model.save(MODELS_DIR / "neural_network.keras")
    joblib.dump(preprocessor, MODELS_DIR / "dl_preprocessor.pkl")
    (MODELS_DIR / "dl_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_deep_learning_model(), indent=2))
