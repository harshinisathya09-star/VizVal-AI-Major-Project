from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_engineering import add_features


def comparable_properties(df: pd.DataFrame, record: dict, n: int = 8) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    data = add_features(df.copy(), include_price_features=True)
    user = add_features(pd.DataFrame([record]), include_price_features=False)
    num_cols = ["area_sqft", "bedrooms", "bathrooms", "property_age", "floor", "total_floors"]
    cat_cols = ["location", "property_type", "parking", "furnished"]

    scaler = StandardScaler()
    data_num = scaler.fit_transform(data[num_cols])
    user_num = scaler.transform(user[num_cols])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    data_cat = encoder.fit_transform(data[cat_cols])
    user_cat = encoder.transform(user[cat_cols])

    data_matrix = np.hstack([data_num, data_cat * 1.5])
    user_matrix = np.hstack([user_num, user_cat * 1.5])
    scores = cosine_similarity(user_matrix, data_matrix).ravel()
    result = data.copy()
    result["similarity"] = scores
    result = result.sort_values("similarity", ascending=False).head(n)
    cols = [
        "area_sqft",
        "bedrooms",
        "bathrooms",
        "property_age",
        "location",
        "property_type",
        "price",
        "price_per_sqft",
        "similarity",
    ]
    return result[cols].reset_index(drop=True)


def comparable_summary(comps: pd.DataFrame) -> dict[str, float | None]:
    if comps.empty:
        return {"average_price": None, "median_price": None, "average_price_per_sqft": None}
    return {
        "average_price": float(comps["price"].mean()),
        "median_price": float(comps["price"].median()),
        "average_price_per_sqft": float(comps["price_per_sqft"].mean()),
    }
