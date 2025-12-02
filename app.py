import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Smart Forest Fire Prediction",
    page_icon="🔥",
    layout="centered"
)

st.title("🔥 Smart Forest Fire Prediction System")
st.write("""
Predict the likelihood of a **forest fire** based on environmental and meteorological
conditions using a **Machine Learning model** trained on the Algerian Forest Fire Dataset.
""")

MODEL_PATH = "forest_fire_best_pipeline_streamlit.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

st.sidebar.header("📥 Enter Input Values")

def user_input_features():
    day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=15)
    month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=6)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=30.0)
    rh = st.sidebar.number_input("Relative Humidity (%)", min_value=0, max_value=100, value=50)
    ws = st.sidebar.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0)
    rain = st.sidebar.number_input("Rain (mm)", min_value=0.0, max_value=50.0, value=0.0)

    ffmc = st.sidebar.number_input("FFMC", min_value=0.0, max_value=100.0, value=80.0)
    dmc = st.sidebar.number_input("DMC", min_value=0.0, max_value=300.0, value=20.0)
    dc = st.sidebar.number_input("DC", min_value=0.0, max_value=1000.0, value=100.0)
    isi = st.sidebar.number_input("ISI", min_value=0.0, max_value=30.0, value=5.0)
    bui = st.sidebar.number_input("BUI", min_value=0.0, max_value=200.0, value=20.0)
    fwi = st.sidebar.number_input("FWI", min_value=0.0, max_value=50.0, value=10.0)

    data = {
        "day": day, "month": month,
        "temperature": temperature, "rh": rh, "ws": ws, "rain": rain,
        "ffmc": ffmc, "dmc": dmc, "dc": dc, 
        "isi": isi, "bui": bui, "fwi": fwi
    }
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("🔎 Input Data Preview")
st.write(input_df)

if st.button("Predict Fire Risk 🚨"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"🔥 High Risk of Fire! (Confidence: {probability:.2f})")
    else:
        st.success(f"🌿 No Fire Risk (Confidence: {probability:.2f})")

st.caption("Dataset: Algerian Forest Fires Dataset (UCI Machine Learning Repository)")
