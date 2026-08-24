from __future__ import annotations

import io
import subprocess
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_dataset
from src.database import load_history, save_prediction
from src.explainability import explain_prediction
from src.feature_engineering import add_features
from src.prediction import load_metrics, models_available, predict_property
from src.preprocessing import clean_property_data, validate_dataset
from src.report_generator import generate_pdf_report
from src.utils import format_inr, status_from_prices
from src.valuation import comparable_properties, comparable_summary

st.set_page_config(
    page_title="Real Estate Price Intelligence",
    page_icon="🏠",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_market_data() -> tuple[pd.DataFrame, str, dict]:
    raw, source = load_dataset()
    quality = validate_dataset(raw)
    return add_features(clean_property_data(raw), include_price_features=True), source, quality


def property_form(prefix: str = "main") -> tuple[dict, float | None]:
    data, _, _ = get_market_data()
    locations = sorted(data["location"].dropna().unique())
    property_types = sorted(data["property_type"].dropna().unique())

    st.subheader("PROPERTY DETAILS")
    col1, col2, col3 = st.columns(3)
    with col1:
        location = st.selectbox("Location", locations, key=f"{prefix}_location")
        area_sqft = st.number_input("Area in sq.ft", 250, 20000, 1450, 50, key=f"{prefix}_area")
    with col2:
        property_type = st.selectbox("Property Type", property_types, key=f"{prefix}_type")
        bedrooms = st.number_input("Bedrooms", 1, 12, 3, key=f"{prefix}_bedrooms")
    with col3:
        bathrooms = st.number_input("Bathrooms", 1, 12, 2, key=f"{prefix}_bathrooms")
        asking_lakhs = st.number_input("Asking Price in Lakhs (optional)", 0.0, 1000.0, 0.0, 1.0, key=f"{prefix}_ask")

    st.subheader("PROPERTY CHARACTERISTICS")
    col4, col5, col6, col7, col8 = st.columns(5)
    with col4:
        property_age = st.number_input("Property Age", 0, 80, 5, key=f"{prefix}_age")
    with col5:
        floor = st.number_input("Floor", 0, 100, 4, key=f"{prefix}_floor")
    with col6:
        total_floors = st.number_input("Total Floors", 1, 100, 12, key=f"{prefix}_total_floors")
    with col7:
        parking = st.selectbox("Parking", ["Yes", "No"], key=f"{prefix}_parking")
    with col8:
        furnished = st.selectbox("Furnished", ["Unfurnished", "Semi-Furnished", "Furnished"], key=f"{prefix}_furnished")

    record = {
        "location": location,
        "property_type": property_type,
        "area_sqft": float(area_sqft),
        "bedrooms": int(bedrooms),
        "bathrooms": int(bathrooms),
        "property_age": float(property_age),
        "floor": int(min(floor, total_floors)),
        "total_floors": int(total_floors),
        "parking": parking,
        "furnished": furnished,
    }
    asking_price = asking_lakhs * 100000 if asking_lakhs > 0 else None
    return record, asking_price


def train_missing_models_panel() -> None:
    st.warning("Saved training artifacts are missing. Train the models before production inference.")
    st.code("python3 -m src.train_ml\npython3 -m src.train_dl", language="bash")
    if st.button("Train traditional ML models now"):
        with st.spinner("Training Linear Regression, Random Forest and XGBoost when installed..."):
            completed = subprocess.run(
                [sys.executable, "-m", "src.train_ml"],
                capture_output=True,
                text=True,
                check=False,
            )
        if completed.returncode == 0:
            st.success("Traditional ML models trained. Refresh the page to load the artifacts.")
            st.code(completed.stdout[-2500:])
            st.cache_data.clear()
        else:
            st.error("Training failed. Install requirements.txt and try again.")
            st.code(completed.stderr[-2500:])


def run_prediction(record: dict, asking_price: float | None) -> None:
    data, _, _ = get_market_data()
    result = predict_property(record)
    comps = comparable_properties(data, record)
    comp_summary = comparable_summary(comps)
    valuation_status = status_from_prices(asking_price, result["final_prediction"], comp_summary["average_price"])
    assessment_id = save_prediction(record, result, asking_price, valuation_status)
    factors = explain_prediction(record)

    st.session_state["last_record"] = record
    st.session_state["last_result"] = result
    st.session_state["last_comps"] = comps
    st.session_state["last_summary"] = comp_summary
    st.session_state["last_status"] = valuation_status
    st.session_state["last_factors"] = factors
    st.session_state["last_assessment_id"] = assessment_id

    a, b, c, d = st.columns(4)
    a.metric("Estimated Value", format_inr(result["final_prediction"]))
    b.metric("Price / sq.ft", f"Rs {result['price_per_sqft']:,.0f}")
    c.metric("ML Prediction", format_inr(result["ml_prediction"]))
    d.metric("DL Prediction", format_inr(result["dl_prediction"]) if result["dl_prediction"] else "Not trained")

    st.subheader("VALUATION RANGE")
    st.info(f"Model-estimated valuation range: {format_inr(result['range_low'])} - {format_inr(result['range_high'])}")
    st.subheader("MARKET POSITION")
    st.write(valuation_status)

    st.subheader("WHY THIS VALUE?")
    if factors.empty:
        st.caption("Feature contribution data is unavailable for the current model.")
    else:
        fig = px.bar(factors, x="contribution", y="feature", orientation="h", title="Strongest model influences")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("These features had the strongest influence on the model prediction. They are not causal evidence.")

    report_path = generate_pdf_report(assessment_id, record, result, valuation_status, comp_summary, factors)
    st.download_button(
        "📄 Download Valuation Report",
        data=report_path.read_bytes(),
        file_name=report_path.name,
        mime="application/pdf",
    )


def market_kpis(df: pd.DataFrame) -> None:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Average Price", format_inr(df["price"].mean()))
    k2.metric("Median Price", format_inr(df["price"].median()))
    k3.metric("Avg / sq.ft", f"Rs {df['price_per_sqft'].mean():,.0f}")
    k4.metric("Most Expensive", df.groupby("location")["price"].mean().idxmax())
    k5.metric("Most Affordable", df.groupby("location")["price"].mean().idxmin())
    k6.metric("Properties", f"{len(df):,}")


def show_eda(df: pd.DataFrame) -> None:
    charts = [
        px.histogram(df, x="price", nbins=40, title="Price distribution"),
        px.scatter(df, x="area_sqft", y="price", color="location", title="Area vs Price"),
        px.box(df, x="bedrooms", y="price", title="Bedrooms vs Price"),
        px.box(df, x="bathrooms", y="price", title="Bathrooms vs Price"),
        px.scatter(df, x="property_age", y="price", color="property_type", title="Property Age vs Price"),
        px.bar(df.groupby("location", as_index=False)["price"].mean(), x="location", y="price", title="Location vs Average Price"),
        px.bar(df.groupby("location", as_index=False)["price_per_sqft"].mean(), x="location", y="price_per_sqft", title="Price per sq.ft by Location"),
        px.bar(df.groupby("property_type", as_index=False)["price"].mean(), x="property_type", y="price", title="Property Type vs Average Price"),
        px.box(df, x="location", y="price", title="Price Distribution by Location"),
    ]
    for left, right in zip(charts[::2], charts[1::2]):
        c1, c2 = st.columns(2)
        c1.plotly_chart(left, use_container_width=True)
        c2.plotly_chart(right, use_container_width=True)
    if len(charts) % 2:
        st.plotly_chart(charts[-1], use_container_width=True)
    corr = df.select_dtypes("number").corr()
    st.plotly_chart(px.imshow(corr, text_auto=True, title="Correlation heatmap"), use_container_width=True)


def page_property_valuation() -> None:
    st.title("🏠 Real Estate Price Intelligence")
    st.caption("AI-powered property valuation using Machine Learning + Deep Learning")
    if not models_available():
        train_missing_models_panel()
        return
    record, asking_price = property_form("valuation")
    c1, c2 = st.columns([1, 1])
    with c1:
        predict = st.button("🔮 Predict Property Value", type="primary")
    with c2:
        if st.button("Reset Form"):
            for key in list(st.session_state):
                if key.startswith("valuation_") or key.startswith("last_"):
                    del st.session_state[key]
            st.rerun()
    if predict:
        try:
            run_prediction(record, asking_price)
        except Exception as exc:
            st.error("Prediction failed. Please verify that models are trained and inputs are valid.")
            st.code(str(exc))


def page_market_intelligence(df: pd.DataFrame, quality: dict, source: str) -> None:
    st.title("📊 Market Intelligence")
    st.info(source)
    market_kpis(df)
    st.subheader("Data Quality")
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Rows", quality["rows"])
    q2.metric("Columns", quality["columns"])
    q3.metric("Missing Values", quality["missing_values"])
    q4.metric("Duplicate Rows", quality["duplicate_rows"])
    q5.metric("Invalid Records", quality["invalid_records"])
    with st.expander("Column names and data types"):
        st.write(pd.DataFrame({"column": quality["column_names"], "dtype": [quality["dtypes"][c] for c in quality["column_names"]]}))
    show_eda(df)


def page_model_comparison() -> None:
    st.title("🤖 Model Comparison")
    metrics = load_metrics()
    rows = metrics.get("metrics", [])
    dl_path = "models/dl_metrics.json"
    try:
        import json
        from pathlib import Path

        if Path(dl_path).exists():
            rows.append(json.loads(Path(dl_path).read_text(encoding="utf-8")))
    except Exception:
        pass
    if not rows:
        train_missing_models_panel()
        return
    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df[["Model", "MAE", "RMSE", "R2"]], use_container_width=True)
    best = metrics_df.sort_values("RMSE").iloc[0]["Model"]
    st.success(f"Best validation performer: {best}")
    st.plotly_chart(px.bar(metrics_df, x="Model", y=["MAE", "RMSE"], barmode="group"), use_container_width=True)
    st.caption("Deep learning uses one-hot location encoding unless the real dataset is large enough for meaningful embeddings.")


def page_explainability() -> None:
    st.title("🧠 AI Explainability")
    if not models_available():
        train_missing_models_panel()
        return
    record, _ = property_form("explain")
    if st.button("Explain Prediction"):
        factors = explain_prediction(record)
        st.plotly_chart(px.bar(factors, x="contribution", y="feature", orientation="h"), use_container_width=True)
        st.caption("Feature importance explains model behavior for this estimate; it should not be read as causation.")


def page_what_if() -> None:
    st.title("🔬 WHAT-IF PROPERTY SIMULATOR")
    if not models_available():
        train_missing_models_panel()
        return
    current, _ = property_form("whatif_current")
    st.subheader("Scenario Property")
    scenario = current.copy()
    c1, c2, c3, c4 = st.columns(4)
    scenario["area_sqft"] = c1.slider("Scenario Area", 300, 20000, int(current["area_sqft"]), 50)
    scenario["bedrooms"] = c2.slider("Scenario Bedrooms", 1, 12, current["bedrooms"])
    scenario["bathrooms"] = c3.slider("Scenario Bathrooms", 1, 12, current["bathrooms"])
    scenario["property_age"] = c4.slider("Scenario Age", 0, 80, int(current["property_age"]))
    c5, c6, c7 = st.columns(3)
    scenario["floor"] = c5.slider("Scenario Floor", 0, int(current["total_floors"]), current["floor"])
    scenario["parking"] = c6.selectbox("Scenario Parking", ["Yes", "No"], index=0 if current["parking"] == "Yes" else 1)
    scenario["furnished"] = c7.selectbox("Scenario Furnished", ["Unfurnished", "Semi-Furnished", "Furnished"])
    if st.button("Run Scenario", type="primary"):
        current_pred = predict_property(current)["final_prediction"]
        scenario_pred = predict_property(scenario)["final_prediction"]
        diff = scenario_pred - current_pred
        a, b, c = st.columns(3)
        a.metric("Current", format_inr(current_pred))
        b.metric("Scenario", format_inr(scenario_pred))
        c.metric("Difference", format_inr(diff), delta=f"{diff / 100000:.2f} L")
        st.caption("Model-based scenario estimate")
        fig = go.Figure(go.Bar(x=["Current Property", "Scenario Property"], y=[current_pred, scenario_pred]))
        st.plotly_chart(fig, use_container_width=True)


def page_comparables(df: pd.DataFrame) -> None:
    st.title("🏘 Comparable Properties")
    record, _ = property_form("comps")
    if st.button("Find Comparable Properties", type="primary"):
        comps = comparable_properties(df, record, n=10)
        summary = comparable_summary(comps)
        c1, c2, c3 = st.columns(3)
        c1.metric("Comparable Average", format_inr(summary["average_price"]))
        c2.metric("Comparable Median", format_inr(summary["median_price"]))
        c3.metric("Average / sq.ft", f"Rs {summary['average_price_per_sqft']:,.0f}" if summary["average_price_per_sqft"] else "N/A")
        st.dataframe(comps, use_container_width=True)


def page_analytics(df: pd.DataFrame) -> None:
    st.title("📈 Analytics")
    market_kpis(df)
    show_eda(df)


def page_history() -> None:
    st.title("🕘 Prediction History")
    history = load_history()
    if history.empty:
        st.info("No predictions have been saved yet.")
        return
    location = st.multiselect("Filter by location", sorted(history["location"].dropna().unique()))
    filtered = history[history["location"].isin(location)] if location else history
    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "Download History CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="prediction_history.csv",
        mime="text/csv",
    )


def page_about() -> None:
    st.title("ℹ️ About")
    st.write(
        "Real Estate Price Intelligence & Property Valuation System is a Streamlit "
        "application for property valuation, model comparison, explainability, comparable "
        "analysis, what-if simulation, SQLite history, and PDF report generation."
    )
    st.write("Training is separated from inference: run `python3 -m src.train_ml` and `python3 -m src.train_dl`.")
    st.write("The fallback synthetic dataset is stored separately under `data/processed` when no CSV is found in `data/raw`.")


def main() -> None:
    df, source, quality = get_market_data()
    st.sidebar.title("Real Estate Intelligence")
    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Property Valuation",
            "📊 Market Intelligence",
            "🤖 Model Comparison",
            "🧠 AI Explainability",
            "🔬 What-If Simulator",
            "🏘 Comparable Properties",
            "📈 Analytics",
            "🕘 Prediction History",
            "ℹ️ About",
        ],
    )
    if page == "🏠 Property Valuation":
        page_property_valuation()
    elif page == "📊 Market Intelligence":
        page_market_intelligence(df, quality, source)
    elif page == "🤖 Model Comparison":
        page_model_comparison()
    elif page == "🧠 AI Explainability":
        page_explainability()
    elif page == "🔬 What-If Simulator":
        page_what_if()
    elif page == "🏘 Comparable Properties":
        page_comparables(df)
    elif page == "📈 Analytics":
        page_analytics(df)
    elif page == "🕘 Prediction History":
        page_history()
    else:
        page_about()


if __name__ == "__main__":
    main()
