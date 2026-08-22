import streamlit as st
import pandas as pd
import importlib.util

st.write("JOBLIB CHECK:", importlib.util.find_spec("joblib"))
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="VizVal AI",
    page_icon="🏠",
    layout="wide"
)

# --------------------------------------------------
# LOAD ML MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("vizag_property_price_model.pkl")


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🏠 Real Estate")

st.sidebar.write("### Property Valuation System")

st.sidebar.info(
    "Enter property details to get an estimated property valuation."
)

st.sidebar.divider()

st.sidebar.write("### 📌 Project Features")

st.sidebar.write("🏘️ Property Details")
st.sidebar.write("📊 ML Property Valuation")
st.sidebar.write("✨ Easy to Use Interface")

st.sidebar.divider()

st.sidebar.subheader("🧭 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🏘️ Property Valuation", "ℹ️ About Project"]
)


# ==================================================
# HOME PAGE
# ==================================================

if page == "🏠 Home":

    st.title("🏠 Welcome to VizVal AI")

    st.subheader(
        "Real Estate Price Intelligence & Property Valuation System"
    )

    st.write("""
    VizVal AI is a machine-learning based property valuation
    system designed to estimate real estate prices using
    property characteristics and location information.
    """)

    st.divider()

    st.header("🎯 Project Objective")

    st.write("""
    The objective of this project is to estimate the market value
    of a property based on factors such as location, property type,
    bedrooms, bathrooms and area.
    """)

    st.divider()

    st.header("✨ Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🏘️ Property Details")
        st.write("""
        Enter location, property type, bedrooms,
        bathrooms and property area.
        """)

    with col2:
        st.subheader("🤖 ML Valuation")
        st.write("""
        The trained machine learning model predicts
        the estimated property value.
        """)

    with col3:
        st.subheader("📊 Price Intelligence")
        st.write("""
        View estimated property value, value in lakhs
        and price per square foot.
        """)

    st.divider()

    st.header("⚙️ How It Works")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.subheader("1️⃣ Enter Details")
        st.write("Provide the property information.")

    with step2:
        st.subheader("2️⃣ ML Processing")
        st.write("The trained ML model processes the inputs.")

    with step3:
        st.subheader("3️⃣ Get Valuation")
        st.write("Receive the estimated property price.")


# ==================================================
# PROPERTY VALUATION PAGE
# ==================================================

elif page == "🏘️ Property Valuation":

    st.title("🏠 Property Valuation")

    st.write(
        "Enter the property details below to estimate its market value."
    )

    st.divider()

    # --------------------------------------------------
    # PROPERTY DETAILS
    # --------------------------------------------------

    st.header("📋 Property Details")

    col1, col2 = st.columns(2)

    with col1:

        location = st.selectbox(
            "📍 Select Location",
            [
                "Visakhapatnam",
                "Hyderabad",
                "Vijayawada",
                "Guntur"
            ]
        )

        neighborhood = st.text_input(
            "📌 Locality",
            "Bheemili"
        )

        bedrooms = st.number_input(
            "🛏️ Number of Bedrooms",
            min_value=1,
            max_value=10,
            value=3
        )

        area = st.number_input(
            "📐 Area (sq.ft)",
            min_value=300,
            max_value=20000,
            value=1500
        )

    with col2:

        property_type = st.selectbox(
            "🏢 Property Type",
            [
                "Villa",
                "Apartment",
                "Independent House",
                "Plot"
            ]
        )

        bathrooms = st.number_input(
            "🛁 Number of Bathrooms",
            min_value=1,
            max_value=10,
            value=3
        )

        property_date = st.date_input(
            "📅 Property Date",
            datetime.today()
        )

    # --------------------------------------------------
    # AMENITIES
    # --------------------------------------------------

    st.header("✨ Amenities")

    amenity_col1, amenity_col2 = st.columns(2)

    with amenity_col1:
        parking = st.checkbox("🚗 Parking")
        lift = st.checkbox("🛗 Lift")

    with amenity_col2:
        swimming_pool = st.checkbox("🏊 Swimming Pool")
        security = st.checkbox("🔒 24/7 Security")

    st.divider()

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    if st.button(
        "🔮 Get Property Valuation",
        type="primary"
    ):

        try:

            # Load trained model
            model = load_model()

            # IMPORTANT:
            # Your training data used "size" as an object/string
            # column, so convert the numeric area into a string.
            size_input = f"{area:.0f} sqft"

            input_data = pd.DataFrame({
                "city": [location],
                "date": [str(property_date)],
                "size": [size_input],
                "type": [property_type],
                "beds": [bedrooms],
                "baths": [bathrooms],
                "neighborhood": [neighborhood]
            })

            # Prediction
            prediction = model.predict(input_data)[0]

            price_per_sqft = prediction / area

            # --------------------------------------------------
            # RESULTS
            # --------------------------------------------------

            st.success(
                "✅ Property valuation completed successfully!"
            )

            st.divider()

            result1, result2, result3 = st.columns(3)

            with result1:
                st.metric(
                    "💰 Estimated Property Value",
                    f"₹{prediction:,.0f}"
                )

            with result2:
                st.metric(
                    "💵 Value in Lakhs",
                    f"₹{prediction / 100000:.2f} L"
                )

            with result3:
                st.metric(
                    "📐 Price / Sq.ft",
                    f"₹{price_per_sqft:,.0f}"
                )

            st.divider()

            st.header("🏠 Property Summary")

            summary1, summary2 = st.columns(2)

            with summary1:

                st.write(
                    f"📍 **Location:** {location}"
                )

                st.write(
                    f"📌 **Locality:** {neighborhood}"
                )

                st.write(
                    f"🏢 **Property Type:** {property_type}"
                )

                st.write(
                    f"📐 **Area:** {area:,.0f} sq.ft"
                )

            with summary2:

                st.write(
                    f"🛏️ **Bedrooms:** {bedrooms}"
                )

                st.write(
                    f"🛁 **Bathrooms:** {bathrooms}"
                )

                st.write(
                    f"🚗 **Parking:** {'Yes' if parking else 'No'}"
                )

                st.write(
                    f"🛗 **Lift:** {'Yes' if lift else 'No'}"
                )

        except Exception as e:

            st.error("❌ Model prediction failed.")

            st.warning(
                "The Streamlit interface is working, but the saved "
                "ML model could not be loaded."
            )

            st.code(str(e))


# ==================================================
# ABOUT PROJECT
# ==================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About VizVal AI")

    st.header(
        "🏠 Real Estate Price Intelligence & Property Valuation System"
    )

    st.write("""
    VizVal AI is a machine learning based real estate
    property valuation system.

    The system allows users to enter property information
    and generates an estimated property value using a
    trained machine learning model.
    """)

    st.header("🎯 Project Objective")

    st.write("""
    The objective is to provide an intelligent and
    user-friendly platform for estimating real estate
    property prices.
    """)

    st.header("🤖 Machine Learning")

    st.write("""
    Multiple regression algorithms were evaluated during
    model development and the best-performing model was
    selected for property price prediction.
    """)

    st.header("💻 Technologies Used")

    st.write("""
    • Python
    • Pandas
    • NumPy
    • Scikit-learn
    • XGBoost
    • Joblib
    • Streamlit
    • Machine Learning
    """)
