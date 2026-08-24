from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from .utils import DATABASE_DIR, ensure_directories

DB_PATH = DATABASE_DIR / "real_estate.db"


def get_connection() -> sqlite3.Connection:
    ensure_directories()
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                location TEXT,
                property_type TEXT,
                area REAL,
                bedrooms INTEGER,
                bathrooms INTEGER,
                property_age REAL,
                floor INTEGER,
                parking TEXT,
                furnished TEXT,
                ml_prediction REAL,
                dl_prediction REAL,
                final_prediction REAL,
                asking_price REAL,
                valuation_status TEXT
            )
            """
        )


def save_prediction(record: dict, result: dict, asking_price: float | None, valuation_status: str) -> int:
    init_db()
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO prediction_history (
                timestamp, location, property_type, area, bedrooms, bathrooms,
                property_age, floor, parking, furnished, ml_prediction, dl_prediction,
                final_prediction, asking_price, valuation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                record.get("location"),
                record.get("property_type"),
                record.get("area_sqft"),
                record.get("bedrooms"),
                record.get("bathrooms"),
                record.get("property_age"),
                record.get("floor"),
                record.get("parking"),
                record.get("furnished"),
                result.get("ml_prediction"),
                result.get("dl_prediction"),
                result.get("final_prediction"),
                asking_price,
                valuation_status,
            ),
        )
        return int(cur.lastrowid)


def load_history() -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM prediction_history ORDER BY prediction_id DESC",
            conn,
        )
