from __future__ import annotations

from pathlib import Path

RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
DATABASE_DIR = ROOT_DIR / "database"
REPORTS_DIR = ROOT_DIR / "reports"

PRICE_TARGET = "price"

NUMERIC_FEATURES = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "property_age",
    "floor",
    "total_floors",
    "total_rooms",
    "bathroom_bedroom_ratio",
    "floor_ratio",
]

CATEGORICAL_FEATURES = [
    "parking",
    "furnished",
    "property_type",
    "location",
    "property_age_group",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def ensure_directories() -> None:
    for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, DATABASE_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def format_inr(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    if value >= 10_000_000:
        return f"Rs {value / 10_000_000:.2f} Cr"
    return f"Rs {value / 100_000:.2f} Lakhs"


def status_from_prices(asking_price: float | None, fair_value: float, comparable_avg: float | None) -> str:
    if not asking_price or asking_price <= 0:
        return "No asking price provided"
    anchor = fair_value
    if comparable_avg and comparable_avg > 0:
        anchor = (fair_value + comparable_avg) / 2
    gap = (asking_price - anchor) / anchor
    if gap > 0.10:
        return "Potentially overpriced based on model and selected comparable properties."
    if gap < -0.10:
        return "Potentially underpriced based on model and selected comparable properties."
    return "Fairly priced relative to the model and selected comparable properties."
