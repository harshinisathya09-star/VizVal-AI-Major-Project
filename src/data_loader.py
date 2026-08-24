from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import DATA_PROCESSED_DIR, DATA_RAW_DIR, RANDOM_STATE, ensure_directories


COLUMN_ALIASES = {
    "area": "area_sqft",
    "sqft": "area_sqft",
    "size": "area_sqft",
    "built_up_area": "area_sqft",
    "bed": "bedrooms",
    "beds": "bedrooms",
    "bhk": "bedrooms",
    "bath": "bathrooms",
    "baths": "bathrooms",
    "age": "property_age",
    "propertyage": "property_age",
    "total_floor": "total_floors",
    "floors": "total_floors",
    "car_parking": "parking",
    "furnishing": "furnished",
    "type": "property_type",
    "city": "location",
    "locality": "location",
    "neighborhood": "location",
    "selling_price": "price",
    "amount": "price",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_").replace("-", "_")
        renamed[col] = COLUMN_ALIASES.get(key, key)
    return df.rename(columns=renamed)


def generate_synthetic_data(n_rows: int = 1500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    locations = {
        "Madhapur": 7800,
        "Gachibowli": 8200,
        "Jubilee Hills": 13200,
        "Kukatpally": 6100,
        "Whitefield": 7600,
        "Indiranagar": 11800,
        "Banjara Hills": 12500,
        "Miyapur": 5600,
        "Electronic City": 5200,
        "Vizag Beach Road": 9000,
    }
    property_types = {
        "Apartment": 1.00,
        "Villa": 1.38,
        "Independent House": 1.18,
        "Studio": 0.82,
        "Plot": 0.72,
    }
    furnished_options = {"Unfurnished": 0, "Semi-Furnished": 275_000, "Furnished": 650_000}

    location = rng.choice(list(locations), n_rows)
    property_type = rng.choice(list(property_types), n_rows, p=[0.48, 0.13, 0.22, 0.09, 0.08])
    bedrooms = rng.integers(1, 6, n_rows)
    bathrooms = np.maximum(1, bedrooms + rng.integers(-1, 2, n_rows))
    area = np.clip(rng.normal(650 + bedrooms * 360, 230, n_rows), 350, 5200).round()
    property_age = np.clip(rng.gamma(2.2, 4.5, n_rows), 0, 35).round()
    total_floors = rng.integers(1, 31, n_rows)
    floor = np.array([rng.integers(0, max(1, tf) + 1) for tf in total_floors])
    parking = np.where(rng.random(n_rows) > 0.28, "Yes", "No")
    furnished = rng.choice(list(furnished_options), n_rows, p=[0.37, 0.43, 0.20])

    base_rate = np.array([locations[x] for x in location])
    type_factor = np.array([property_types[x] for x in property_type])
    furnished_premium = np.array([furnished_options[x] for x in furnished])
    age_discount = np.maximum(0.72, 1 - property_age * 0.012)
    floor_factor = 1 + np.minimum(floor / np.maximum(total_floors, 1), 1) * 0.035
    room_premium = (bedrooms - 2) * 115_000 + (bathrooms - 2) * 75_000
    parking_premium = np.where(parking == "Yes", 320_000, -90_000)
    nonlinear_area = np.power(area, 1.035) * base_rate * type_factor
    location_noise = rng.normal(1.0, 0.075, n_rows)
    price = nonlinear_area * age_discount * floor_factor * location_noise
    price = price + furnished_premium + room_premium + parking_premium + rng.normal(0, 450_000, n_rows)
    price = np.clip(price, 1_500_000, None).round()

    df = pd.DataFrame(
        {
            "area_sqft": area.astype(int),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "property_age": property_age.astype(int),
            "floor": floor,
            "total_floors": total_floors,
            "parking": parking,
            "furnished": furnished,
            "property_type": property_type,
            "location": location,
            "price": price.astype(int),
        }
    )
    return df


def load_dataset(csv_path: str | Path | None = None) -> tuple[pd.DataFrame, str]:
    ensure_directories()
    if csv_path:
        path = Path(csv_path)
        if path.exists():
            return normalize_columns(pd.read_csv(path)), f"Loaded dataset from {path}"

    candidates = sorted(DATA_RAW_DIR.glob("*.csv"))
    if candidates:
        return normalize_columns(pd.read_csv(candidates[0])), f"Loaded dataset from {candidates[0]}"

    df = generate_synthetic_data()
    fallback_path = DATA_PROCESSED_DIR / "synthetic_real_estate.csv"
    df.to_csv(fallback_path, index=False)
    return df, "No CSV found in data/raw, using separated synthetic demonstration data."
