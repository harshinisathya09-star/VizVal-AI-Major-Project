from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, PRICE_TARGET


def clean_property_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    required_defaults = {
        "area_sqft": 1000,
        "bedrooms": 2,
        "bathrooms": 2,
        "property_age": 5,
        "floor": 1,
        "total_floors": 5,
        "parking": "No",
        "furnished": "Unfurnished",
        "property_type": "Apartment",
        "location": "Unknown",
    }
    for col, default in required_defaults.items():
        if col not in cleaned.columns:
            cleaned[col] = default

    for col in ["area_sqft", "bedrooms", "bathrooms", "property_age", "floor", "total_floors", PRICE_TARGET]:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    for col in ["parking", "furnished", "property_type", "location"]:
        cleaned[col] = cleaned[col].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"})

    cleaned["area_sqft"] = cleaned["area_sqft"].clip(lower=200, upper=20000)
    cleaned["bedrooms"] = cleaned["bedrooms"].clip(lower=1, upper=12)
    cleaned["bathrooms"] = cleaned["bathrooms"].clip(lower=1, upper=12)
    cleaned["property_age"] = cleaned["property_age"].clip(lower=0, upper=80)
    cleaned["total_floors"] = cleaned["total_floors"].clip(lower=1, upper=100)
    cleaned["floor"] = np.minimum(cleaned["floor"].clip(lower=0, upper=100), cleaned["total_floors"])
    if PRICE_TARGET in cleaned.columns:
        cleaned = cleaned[cleaned[PRICE_TARGET].notna() & (cleaned[PRICE_TARGET] > 100_000)]
    return cleaned.drop_duplicates().reset_index(drop=True)


def validate_dataset(df: pd.DataFrame) -> dict[str, object]:
    numeric = df.select_dtypes(include=[np.number])
    invalid_mask = pd.Series(False, index=df.index)
    checks = {
        "negative_area": "area_sqft" in df and (df["area_sqft"] <= 0),
        "impossible_bedrooms": "bedrooms" in df and ~df["bedrooms"].between(1, 12),
        "impossible_bathrooms": "bathrooms" in df and ~df["bathrooms"].between(1, 12),
        "invalid_prices": PRICE_TARGET in df and (df[PRICE_TARGET] <= 0),
    }
    for value in checks.values():
        if not isinstance(value, bool):
            invalid_mask = invalid_mask | value.fillna(False)
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "invalid_records": int(invalid_mask.sum()),
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_summary": numeric.describe().T if not numeric.empty else pd.DataFrame(),
    }


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[MODEL_FEATURES].copy()
    y = df[PRICE_TARGET].copy()
    return X, y
