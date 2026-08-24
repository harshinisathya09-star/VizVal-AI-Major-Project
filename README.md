# Real Estate Price Intelligence & Property Valuation System

AI-powered property valuation using Machine Learning and Deep Learning.

## Features

- Streamlit dashboard with valuation, market intelligence, EDA, comparables, what-if simulator, explainability, history, and reports.
- Traditional ML training: Linear Regression, Random Forest, and XGBoost.
- Deep Learning training: TensorFlow/Keras regression network.
- SQLite prediction history in `database/real_estate.db`.
- PDF valuation reports in `reports/`.
- Synthetic data fallback when no CSV exists in `data/raw/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train Models

Place a CSV in `data/raw/` with columns similar to:

`area_sqft, bedrooms, bathrooms, property_age, floor, total_floors, parking, furnished, property_type, location, price`

Then run:

```bash
python3 -m src.train_ml
python3 -m src.train_dl
```

If no CSV is found, the training scripts create a realistic synthetic demonstration dataset at `data/processed/synthetic_real_estate.csv`.

## Run App

```bash
streamlit run app.py
```

The app loads saved models for inference and does not retrain on startup.
