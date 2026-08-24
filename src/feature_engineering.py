from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame, include_price_features: bool = True) -> pd.DataFrame:
    featured = df.copy()
    featured["total_rooms"] = featured["bedrooms"] + featured["bathrooms"]
    featured["bathroom_bedroom_ratio"] = featured["bathrooms"] / featured["bedrooms"].replace(0, np.nan)
    featured["bathroom_bedroom_ratio"] = featured["bathroom_bedroom_ratio"].fillna(1.0).clip(0.2, 5)
    featured["floor_ratio"] = featured["floor"] / featured["total_floors"].replace(0, np.nan)
    featured["floor_ratio"] = featured["floor_ratio"].fillna(0).clip(0, 1)
    featured["property_age_group"] = pd.cut(
        featured["property_age"],
        bins=[-1, 2, 8, 20, 100],
        labels=["New", "Recent", "Established", "Older"],
    ).astype(str)
    if include_price_features and "price" in featured.columns:
        featured["price_per_sqft"] = featured["price"] / featured["area_sqft"].replace(0, np.nan)
    return featured
