import os
import joblib
import pandas as pd
import streamlit as st
from textwrap import dedent


def render_html(content):
    content = dedent(content).strip()

    if hasattr(st, "html"):
        st.html(content)
    else:
        st.markdown(content, unsafe_allow_html=True)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AirSense AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "aqi_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "cleaned_air_quality.csv"
)


# ============================================================
# LOAD MODEL + DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
df = load_data()


# ============================================================
# SESSION STATE
# ============================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "category" not in st.session_state:
    st.session_state.category = None


# ============================================================
# AQI FUNCTIONS
# ============================================================

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

        "Good":
            "Air quality is considered safe with little or no health concern.",

        "Satisfactory":
            "Air quality is acceptable, although sensitive people may need some caution.",

        "Moderate":
            "Sensitive individuals may experience discomfort during prolonged exposure.",

        "Poor":
            "Prolonged exposure may cause health effects. Consider reducing outdoor exposure.",

        "Very Poor":
            "Health alert: prolonged exposure may affect everyone.",

        "Severe":
            "Serious health effects are possible. Avoid prolonged exposure."
    }

    return descriptions[category]


def get_category_emoji(category):

    emojis = {

        "Good": "☀️",

        "Satisfactory": "🌤️",

        "Moderate": "⛅",

        "Poor": "🌫️",

        "Very Poor": "🌁",

        "Severe": "⚠️"
    }

    return emojis[category]


def get_category_colors(category):

    colors = {

        "Good": {
            "accent": "#16a34a",
            "text": "#166534"
        },

        "Satisfactory": {
            "accent": "#eab308",
            "text": "#854d0e"
        },

        "Moderate": {
            "accent": "#f97316",
            "text": "#9a3412"
        },

        "Poor": {
            "accent": "#ef4444",
            "text": "#991b1b"
        },

        "Very Poor": {
            "accent": "#9333ea",
            "text": "#6b21a8"
        },

        "Severe": {
            "accent": "#374151",
            "text": "#111827"
        }
    }

    return colors[category]


# ============================================================
# CUSTOM CSS — AIRSENSE DAY / NIGHT SKY
# ============================================================

# Appearance selector
with st.sidebar:
    appearance = st.radio(
        "🌈 APPEARANCE",
        ["☀️ Light", "🌙 Dark", "🖥️ System"],
        index=0
    )

if appearance == "🌙 Dark":
    mode = "dark"
elif appearance == "🖥️ System":
    mode = "system"
else:
    mode = "light"


# ------------------------------------------------------------
# COLORS
# ------------------------------------------------------------

if mode == "dark":
    SKY1 = "#020617"
    SKY2 = "#0b1d36"
    SKY3 = "#102a43"

    TEXT = "#f8fafc"
    MUTED = "#cbd5e1"

    CARD = "rgba(15,23,42,.78)"
    INPUT = "rgba(15,23,42,.95)"

    CITY1 = "#07111f"
    CITY2 = "#10233b"

elif mode == "light":
    SKY1 = "#38bdf8"
    SKY2 = "#7dd3fc"
    SKY3 = "#e0f2fe"

    TEXT = "#10213f"
    MUTED = "#475569"

    CARD = "rgba(255,255,255,.72)"
    INPUT = "rgba(255,255,255,.90)"

    CITY1 = "#38576a"
    CITY2 = "#547383"

else:
    SKY1 = "#38bdf8"
    SKY2 = "#7dd3fc"
    SKY3 = "#e0f2fe"

    TEXT = "#10213f"
    MUTED = "#475569"

    CARD = "rgba(255,255,255,.72)"
    INPUT = "rgba(255,255,255,.90)"

    CITY1 = "#38576a"
    CITY2 = "#547383"


# ------------------------------------------------------------
# MAIN SKY
# ------------------------------------------------------------

render_html(f"""
<style>

/* REMOVE DEFAULT STREAMLIT BACKGROUND */

html,
body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
    background: transparent !important;
}}


/* ==========================================================
   FULL SCREEN SKY
   ========================================================== */

[data-testid="stApp"] {{
    min-height: 100vh !important;
    overflow-x: hidden !important;
}}

[data-testid="stApp"]::before {{

    content: "";

    position: fixed;

    left: 0;
    top: 0;

    width: 100vw;
    height: 100vh;

    z-index: 0;

    pointer-events: none;

    background:
        linear-gradient(
            180deg,
            {SKY1} 0%,
            {SKY2} 48%,
            {SKY3} 100%
        );
}}


/* ==========================================================
   SUN / MOON
   ========================================================== */

[data-testid="stAppViewContainer"]::before {{

    content: {"'🌙'" if mode == "dark" else "'☀️'"};

    position: fixed;

    top: 55px;
    right: 90px;

    width: 120px;
    height: 120px;

    z-index: 1;

    display: flex;

    justify-content: center;
    align-items: center;

    font-size: 78px;

    pointer-events: none;

    animation:
        floatingSun 6s ease-in-out infinite;

}}

@keyframes floatingSun {{

    0%,100% {{
        transform: translateY(0);
    }}

    50% {{
        transform: translateY(15px);
    }}

}}


/* ==========================================================
   MOVING CLOUDS
   ========================================================== */

[data-testid="stMain"]::before {{

    content:
        "☁️        ☁️              ☁️          ☁️";

    position: fixed;

    top: 15vh;
    left: -400px;

    width: 200vw;

    z-index: 1;

    white-space: nowrap;

    font-size: 60px;

    letter-spacing: 50px;

    opacity:
        {"0.20" if mode == "dark" else "0.80"};

    pointer-events: none;

    animation:
        cloudMove 45s linear infinite;

}}

@keyframes cloudMove {{

    from {{
        transform: translateX(0);
    }}

    to {{
        transform: translateX(100vw);
    }}

}}


/* ==========================================================
   SECOND CLOUD LAYER
   ========================================================== */

[data-testid="stMain"]::after {{

    content:
        "☁️              ☁️                 ☁️";

    position: fixed;

    top: 32vh;

    left: -500px;

    width: 200vw;

    z-index: 1;

    white-space: nowrap;

    font-size: 42px;

    letter-spacing: 90px;

    opacity:
        {"0.10" if mode == "dark" else "0.45"};

    pointer-events: none;

    animation:
        cloudMove2 65s linear infinite;

}}

@keyframes cloudMove2 {{

    from {{
        transform: translateX(0);
    }}

    to {{
        transform: translateX(110vw);
    }}

}}


/* ==========================================================
   CITY AT BOTTOM
   ========================================================== */

[data-testid="stApp"]::after {{

    content: "";

    position: fixed;

    left: 0;
    bottom: 0;

    width: 100vw;

    height: 190px;

    z-index: 2;

    pointer-events: none;

    background:

        linear-gradient({CITY1},{CITY1})
        0% 100% / 7% 55% no-repeat,

        linear-gradient({CITY2},{CITY2})
        8% 100% / 8% 75% no-repeat,

        linear-gradient({CITY1},{CITY1})
        17% 100% / 6% 45% no-repeat,

        linear-gradient({CITY2},{CITY2})
        24% 100% / 10% 85% no-repeat,

        linear-gradient({CITY1},{CITY1})
        35% 100% / 8% 60% no-repeat,

        linear-gradient({CITY2},{CITY2})
        44% 100% / 12% 92% no-repeat,

        linear-gradient({CITY1},{CITY1})
        57% 100% / 7% 50% no-repeat,

        linear-gradient({CITY2},{CITY2})
        65% 100% / 10% 78% no-repeat,

        linear-gradient({CITY1},{CITY1})
        77% 100% / 8% 58% no-repeat,

        linear-gradient({CITY2},{CITY2})
        86% 100% / 9% 88% no-repeat,

        linear-gradient({CITY1},{CITY1})
        96% 100% / 6% 65% no-repeat;

}}


/* ==========================================================
   TREES
   ========================================================== */

[data-testid="stAppViewContainer"]::after {{

    content:
        "🌳    🌲       🌳       🌲        🌳       🌲       🌳";

    position: fixed;

    bottom: 115px;

    left: -20px;

    width: 110vw;

    z-index: 3;

    font-size: 43px;

    letter-spacing: 15px;

    white-space: nowrap;

    pointer-events: none;

}}


/* ==========================================================
   DASHBOARD ABOVE BACKGROUND
   ========================================================== */

[data-testid="stMainBlockContainer"] {{

    position: relative !important;

    z-index: 10 !important;

    background: transparent !important;

    padding-bottom: 230px !important;

}}


/* ==========================================================
   TEXT
   ========================================================== */

[data-testid="stMainBlockContainer"] p,
[data-testid="stMainBlockContainer"] h1,
[data-testid="stMainBlockContainer"] h2,
[data-testid="stMainBlockContainer"] h3,
[data-testid="stMainBlockContainer"] h4,
[data-testid="stMainBlockContainer"] label,
[data-testid="stMetricValue"] {{

    color: {TEXT} !important;

}}

[data-testid="stMetricLabel"] {{

    color: {MUTED} !important;

}}


/* ==========================================================
   GLASS CARDS
   ========================================================== */

.glass-card,
.aqi-card {{

    background: {CARD} !important;

    color: {TEXT} !important;

    border:
        1px solid rgba(255,255,255,.30) !important;

    backdrop-filter: blur(16px);

    -webkit-backdrop-filter: blur(16px);

    box-shadow:
        0 12px 35px rgba(0,0,0,.12);

}}

.glass-card h2,
.glass-card h3,
.glass-card p {{

    color: {TEXT} !important;

}}


/* ==========================================================
   INPUTS
   ========================================================== */

.stNumberInput input {{

    color: {TEXT} !important;

    background: {INPUT} !important;

}}


/* ==========================================================
   SIDEBAR
   ========================================================== */

[data-testid="stSidebar"] {{

    background:
        linear-gradient(
            180deg,
            #081226,
            #172b5b,
            #081226
        ) !important;

    z-index: 100 !important;

}}

[data-testid="stSidebar"] * {{

    color: #f8fafc !important;

}}


/* ==========================================================
   BUTTON
   ========================================================== */

.stButton > button {{

    color: white !important;

    background:
        linear-gradient(
            90deg,
            #2563eb,
            #0ea5e9
        ) !important;

    border: none !important;

    border-radius: 14px !important;

    font-weight: 700 !important;

}}

</style>
""")


# ============================================================
# SYSTEM DARK MODE
# ============================================================

if mode == "system":

    render_html("""
    <style>

    @media (prefers-color-scheme: dark) {

        [data-testid="stApp"]::before {

            background:
                linear-gradient(
                    180deg,
                    #020617 0%,
                    #0b1d36 50%,
                    #102a43 100%
                ) !important;

        }

        [data-testid="stAppViewContainer"]::before {

            content: "🌙" !important;

        }

        [data-testid="stMainBlockContainer"] p,
        [data-testid="stMainBlockContainer"] h1,
        [data-testid="stMainBlockContainer"] h2,
        [data-testid="stMainBlockContainer"] h3,
        [data-testid="stMainBlockContainer"] label,
        [data-testid="stMetricValue"] {

            color: #f8fafc !important;

        }

    }

    </style>
    """)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        dedent("""
        <div class="sidebar-logo">

            <div class="sidebar-icon">
                🌍
            </div>

            <h2>
                AirSense AI
            </h2>

            <p>
                Smart Air Quality Intelligence
            </p>

        </div>
        """),
    )


    page = st.radio(
        "NAVIGATION",
        [
            "🔮 AQI Prediction",
            "📊 Data Analysis",
            "📈 Pollution Trends",
            "🤖 Model Insights"
        ]
    )


    render_html(
        dedent("""
        <div class="status-card">

            <div class="status-title">
                System Status
            </div>

            <div class="status-line">
                <span class="online-dot"></span>
                ML Model Online
            </div>

            <div class="status-line">
                🗃️ Dataset Loaded
            </div>

            <div class="status-line">
                🔮 Prediction Ready
            </div>

        </div>
        """),
    )


    render_html(
        dedent("""
        <div class="sidebar-footer">

            AirSense AI

            <br>

            ML + Streamlit

        </div>
        """),
    )


# ============================================================
# HERO SECTION
# ============================================================

render_html(
    dedent("""
    <div class="hero">

        <div class="sun"></div>

        <div class="cloud cloud-one">
            ☁️
        </div>

        <div class="cloud cloud-two">
            ☁️
        </div>

        <h1>
            AirSense AI
        </h1>

        <p>
            Intelligent air-quality prediction and pollution
            analysis powered by Machine Learning.
        </p>

    </div>
    """),
)


# ============================================================
# DATASET METRICS
# ============================================================

avg_aqi = df["aqi_index"].mean()

max_aqi = df["aqi_index"].max()

min_aqi = df["aqi_index"].min()


m1, m2, m3, m4 = st.columns(4)


m1.metric(
    "📊 Total Records",
    f"{len(df):,}"
)


m2.metric(
    "🌫️ Average AQI",
    f"{avg_aqi:.1f}"
)


m3.metric(
    "🔴 Maximum AQI",
    f"{max_aqi:.0f}"
)


m4.metric(
    "🟢 Minimum AQI",
    f"{min_aqi:.0f}"
)


# ============================================================
# PAGE 1 — AQI PREDICTION
# ============================================================

if page == "🔮 AQI Prediction":

    render_html(
        '<div class="section-title">🌤️ Environmental Parameters</div>',
    )


    # --------------------------------------------------------
    # WEATHER INPUTS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        temp = st.number_input(
            "🌡️ Temperature (°C)",
            min_value=0.0,
            max_value=60.0,
            value=25.0
        )


    with c2:

        humidity = st.number_input(
            "💧 Humidity (%)",
            min_value=0,
            max_value=100,
            value=60
        )


    with c3:

        pressure = st.number_input(
            "🌀 Pressure (mb)",
            min_value=800.0,
            max_value=1200.0,
            value=1010.0
        )


    with c4:

        windspeed = st.number_input(
            "💨 Wind Speed (km/h)",
            min_value=0.0,
            max_value=100.0,
            value=10.0
        )


    # --------------------------------------------------------
    # POLLUTION INPUTS
    # --------------------------------------------------------

    render_html(
        '<div class="section-title">🏭 Pollution Parameters</div>',
    )


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


    st.write("")


    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    b1, b2, b3 = st.columns([1, 2, 1])


    with b2:

        predict_clicked = st.button(
            "🔮  PREDICT AIR QUALITY",
            width="stretch"
        )


    if predict_clicked:

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


        prediction = model.predict(
            input_data
        )[0]


        prediction = max(
            0,
            min(500, prediction)
        )


        category = get_aqi_category(
            prediction
        )


        st.session_state.prediction = prediction

        st.session_state.category = category


    # --------------------------------------------------------
    # SHOW RESULT
    # --------------------------------------------------------

    if st.session_state.prediction is not None:

        prediction = st.session_state.prediction

        category = st.session_state.category

        colors = get_category_colors(
            category
        )

        emoji = get_category_emoji(
            category
        )


        # ----------------------------------------------------
        # MAIN AQI CARD
        # ----------------------------------------------------

        render_html(
            dedent(f"""
            <div class="aqi-card"
                 style="border-top: 6px solid {colors['accent']};">

                <div style="
                    font-size:45px;
                    margin-bottom:5px;
                ">
                    {emoji}
                </div>

                <div style="
                    color:#64748b;
                    font-size:14px;
                    text-transform:uppercase;
                    letter-spacing:2px;
                ">
                    YOUR PREDICTED AIR QUALITY
                </div>

                <div class="aqi-number"
                     style="color:{colors['accent']};">

                    {prediction:.1f}

                </div>

                <div class="aqi-label"
                     style="color:{colors['text']};">

                    {category}

                </div>

                <div class="aqi-description">

                    {get_category_description(category)}

                </div>

            </div>
            """),
            )


        # ----------------------------------------------------
        # AQI SCALE
        # ----------------------------------------------------

        marker_position = min(
            100,
            max(
                0,
                (prediction / 500) * 100
            )
        )


        render_html(
            dedent(f"""
            <div class="aqi-scale">

                <div class="scale-bar">

                    <div style="
                        position:absolute;
                        left:{marker_position}%;
                        transform:translateX(-50%);
                        margin-top:-6px;
                        width:24px;
                        height:24px;
                        background:white;
                        border:4px solid #0f172a;
                        border-radius:50%;
                        box-shadow:
                            0 3px 10px
                            rgba(0,0,0,0.25);
                    ">
                    </div>

                </div>

                <div class="scale-labels">

                    <span>Good</span>

                    <span>Satisfactory</span>

                    <span>Moderate</span>

                    <span>Poor</span>

                    <span>Very Poor</span>

                    <span>Severe</span>

                </div>

            </div>
            """),
            )


        # ----------------------------------------------------
        # SUMMARY CARDS
        # ----------------------------------------------------

        r1, r2, r3 = st.columns(3)


        with r1:

            render_html(
                dedent(f"""
                <div class="glass-card">

                    <h3>
                        🌫️ Pollution Level
                    </h3>

                    <h2>
                        {category}
                    </h2>

                    <p>
                        Based on the predicted AQI value.
                    </p>

                </div>
                """),
                    )


        with r2:

            if prediction <= 100:

                risk = "Low"

            elif prediction <= 200:

                risk = "Moderate"

            else:

                risk = "High"


            render_html(
                dedent(f"""
                <div class="glass-card">

                    <h3>
                        ⚠️ Risk Level
                    </h3>

                    <h2>
                        {risk}
                    </h2>

                    <p>
                        Estimated environmental risk.
                    </p>

                </div>
                """),
                    )


        with r3:

            render_html(
                dedent(f"""
                <div class="glass-card">

                    <h3>
                        💡 Air Quality Insight
                    </h3>

                    <p>
                        {get_category_description(category)}
                    </p>

                </div>
                """),
                    )


        # ----------------------------------------------------
        # POLLUTANT CHART
        # ----------------------------------------------------

        render_html(
            '<div class="section-title">📊 Current Input Analysis</div>',
            )


        pollutant_data = pd.DataFrame({

            "Pollutant": [
                "PM2.5",
                "PM10",
                "CO",
                "NO₂"
            ],

            "Value": [
                pm25,
                pm10,
                co,
                no2
            ]

        })


        st.bar_chart(
            pollutant_data.set_index(
                "Pollutant"
            )
        )


# ============================================================
# PAGE 2 — DATA ANALYSIS
# ============================================================

elif page == "📊 Data Analysis":

    render_html(
        '<div class="section-title">📊 Air Quality Data Analysis</div>',
    )


    st.write(
        "Explore the statistical characteristics "
        "of the air-quality dataset."
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


    render_html(
        '<div class="section-title">📋 Pollutant Statistics</div>',
    )


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

        width="stretch"
    )


    render_html(
        '<div class="section-title">📊 AQI Category Distribution</div>',
    )


    category_counts = (
        df["aqi_index"]
        .apply(get_aqi_category)
        .value_counts()
    )


    st.bar_chart(
        category_counts
    )


# ============================================================
# PAGE 3 — POLLUTION TRENDS
# ============================================================

elif page == "📈 Pollution Trends":

    render_html(
        '<div class="section-title">📈 Pollution Trends</div>',
    )


    st.write(
        "Explore how AQI and pollutant levels "
        "change over time."
    )


    trend_df = df.copy()


    trend_df["datetime"] = pd.to_datetime(
        trend_df["datetime"]
    )


    daily_aqi = (
        trend_df
        .set_index("datetime")
        .resample("D")["aqi_index"]
        .mean()
    )


    render_html(
        '<div class="section-title">🌫️ Daily Average AQI</div>',
    )


    st.line_chart(
        daily_aqi
    )


    pollutant_df = (
        trend_df
        .set_index("datetime")
        .resample("D")[
            [
                "pm2_5",
                "pm10",
                "no2",
                "co"
            ]
        ]
        .mean()
    )


    render_html(
        '<div class="section-title">🏭 Pollutant Trends</div>',
    )


    st.line_chart(
        pollutant_df
    )


# ============================================================
# PAGE 4 — MODEL INSIGHTS
# ============================================================

elif page == "🤖 Model Insights":

    render_html(
        '<div class="section-title">🤖 Machine Learning Insights</div>',
    )


    st.write(
        "The machine-learning model uses environmental "
        "and pollutant parameters to predict AQI."
    )


    feature_names = [

        "Temperature",

        "Humidity",

        "Pressure",

        "Wind Speed",

        "PM2.5",

        "PM10",

        "CO",

        "NO₂"

    ]


    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": model.feature_importances_

    })


    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
    )


    render_html(
        '<div class="section-title">🎯 Feature Importance</div>',
    )


    st.bar_chart(
        importance_df.set_index(
            "Feature"
        )
    )


    st.dataframe(
        importance_df.round(4),
        width="stretch"
    )


    st.success(
        "Higher feature importance indicates a stronger "
        "contribution to the model's AQI prediction."
    )


# ============================================================
# FOOTER
# ============================================================

render_html(
    dedent("""
    <div class="footer">

        🌍 <b>AirSense AI</b>

        <br>

        Air Quality Prediction & Pollution Risk Analysis

        <br><br>

        Machine Learning • Python • Streamlit

    </div>
    """),
)