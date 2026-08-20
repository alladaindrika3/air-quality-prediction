import streamlit as st
import pandas as pd
import joblib
import os


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌍",
    layout="wide"
)


# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "..", "air_quality_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "cleaned_air_quality.csv")

# =====================================================
# LOAD MODEL + DATA
# =====================================================

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)


# =====================================================
# FUNCTIONS
# =====================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def get_category_description(category):

    descriptions = {
        "Good": "Air quality is considered safe.",
        "Satisfactory": "Air quality is acceptable.",
        "Moderate": "Sensitive people may experience discomfort.",
        "Poor": "Health effects may occur with prolonged exposure.",
        "Very Poor": "Health alert for everyone.",
        "Severe": "Serious health effects are possible."
    }

    return descriptions[category]


# =====================================================
# HEADER
# =====================================================

st.title("🌍 Air Quality Prediction & Pollution Risk Analysis")

st.markdown(
    """
    **Machine Learning based air-quality monitoring system**

    Predict AQI using weather and pollutant parameters,
    analyze pollution patterns and identify major contributing factors.
    """
)

st.divider()


# =====================================================
# DASHBOARD METRICS
# =====================================================

avg_aqi = df["aqi_index"].mean()
max_aqi = df["aqi_index"].max()
min_aqi = df["aqi_index"].min()

m1, m2, m3, m4 = st.columns(4)

m1.metric("📊 Total Records", f"{len(df):,}")
m2.metric("🌫️ Average AQI", f"{avg_aqi:.1f}")
m3.metric("🔴 Maximum AQI", f"{max_aqi:.0f}")
m4.metric("🟢 Minimum AQI", f"{min_aqi:.0f}")


st.divider()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🌍 Air Quality Dashboard")

page = st.sidebar.radio(
    "Navigate",
    [
        "🔮 AQI Prediction",
        "📊 Data Analysis",
        "📈 Pollution Trends",
        "🤖 Model Insights"
    ]
)


# =====================================================
# PAGE 1 — AQI PREDICTION
# =====================================================

if page == "🔮 AQI Prediction":

    st.header("🔮 Predict Air Quality Index")

    st.write(
        "Enter the current weather and pollutant parameters "
        "to estimate the AQI."
    )

    st.subheader("🌤️ Weather Parameters")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        temp = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=60.0,
            value=25.0
        )

    with c2:
        humidity = st.number_input(
            "Humidity (%)",
            min_value=0,
            max_value=100,
            value=60
        )

    with c3:
        pressure = st.number_input(
            "Pressure (mb)",
            min_value=800.0,
            max_value=1200.0,
            value=1010.0
        )

    with c4:
        windspeed = st.number_input(
            "Wind Speed (km/h)",
            min_value=0.0,
            max_value=100.0,
            value=10.0
        )

    st.subheader("🏭 Pollutant Parameters")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        pm25 = st.number_input(
            "PM2.5",
            min_value=0.0,
            value=50.0
        )

    with c2:
        pm10 = st.number_input(
            "PM10",
            min_value=0.0,
            value=80.0
        )

    with c3:
        co = st.number_input(
            "CO",
            min_value=0,
            value=500
        )

    with c4:
        no2 = st.number_input(
            "NO₂",
            min_value=0.0,
            value=30.0
        )

    if st.button(
        "🔮 Predict AQI",
        use_container_width=True,
        type="primary"
    ):

        input_data = pd.DataFrame({
            "temp_c": [temp],
            "humidity": [humidity],
            "pressure_mb": [pressure],
            "windspeed_kph": [windspeed],
            "pm2_5": [pm25],
            "pm10": [pm10],
            "co": [co],
            "no2": [no2]
        })

        prediction = model.predict(input_data)[0]

        prediction = max(0, min(500, prediction))

        category = get_aqi_category(prediction)

        st.divider()

        r1, r2 = st.columns(2)

        r1.metric(
            "Predicted AQI",
            f"{prediction:.2f}"
        )

        r2.metric(
            "Pollution Category",
            category
        )

        st.info(
            get_category_description(category)
        )


# =====================================================
# PAGE 2 — DATA ANALYSIS
# =====================================================

elif page == "📊 Data Analysis":

    st.header("📊 Air Quality Data Analysis")

    st.write(
        "Statistical overview of weather and pollutant parameters."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average PM2.5",
        f"{df['pm2_5'].mean():.2f}"
    )

    c2.metric(
        "Average PM10",
        f"{df['pm10'].mean():.2f}"
    )

    c3.metric(
        "Average NO₂",
        f"{df['no2'].mean():.2f}"
    )

    st.subheader("Pollutant Statistics")

    st.dataframe(
        df[
            [
                "pm2_5",
                "pm10",
                "co",
                "no2",
                "aqi_index"
            ]
        ].describe().round(2),
        use_container_width=True
    )

    st.subheader("AQI Category Distribution")

    category_counts = df["aqi_index"].apply(
        get_aqi_category
    ).value_counts()

    st.bar_chart(category_counts)


# =====================================================
# PAGE 3 — POLLUTION TRENDS
# =====================================================

elif page == "📈 Pollution Trends":

    st.header("📈 Pollution Trends")

    st.write(
        "Explore AQI and pollutant trends across the dataset."
    )

    df["datetime"] = pd.to_datetime(df["datetime"])

    trend_df = (
        df.set_index("datetime")
        .resample("D")["aqi_index"]
        .mean()
    )

    st.subheader("Daily Average AQI")

    st.line_chart(trend_df)

    st.subheader("Pollutant Trends")

    pollutant_df = (
        df.set_index("datetime")
        .resample("D")[
            ["pm2_5", "pm10", "no2", "co"]
        ]
        .mean()
    )

    st.line_chart(pollutant_df)


# =====================================================
# PAGE 4 — MODEL INSIGHTS
# =====================================================

elif page == "🤖 Model Insights":

    st.header("🤖 Machine Learning Model Insights")

    st.write(
        "The model uses weather and pollutant parameters "
        "to predict AQI."
    )

    feature_names = [
        "Temperature",
        "Humidity",
        "Pressure",
        "Wind Speed",
        "PM2.5",
        "PM10",
        "CO",
        "NO2"
    ]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    st.subheader("Feature Importance")

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.dataframe(
        importance_df.round(4),
        use_container_width=True
    )

    st.success(
        "Higher feature importance indicates a stronger "
        "contribution to the model's AQI prediction."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Air Quality Prediction & Pollution Risk Analysis System | "
    "Machine Learning + Streamlit"
)