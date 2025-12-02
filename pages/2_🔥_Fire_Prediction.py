import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Fire Prediction",
    page_icon="🔥",
    layout="wide"
)

# ----------------------------------------------------------
# LOAD DATA + MODEL
# ----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Algerian_forest_fires_dataset.csv")
    df.columns = df.columns.str.lower().str.strip()
    return df

@st.cache_resource
def load_model():
    return joblib.load("forest_fire_best_pipeline.pkl")

df = load_data()
model = load_model()

st.title("🔥 Forest Fire Prediction System")
st.markdown("Enter environmental parameters below to predict **Fire / No Fire**")

# ----------------------------------------------------------
# CLEANING
# ----------------------------------------------------------
# Only keep numeric columns for sliders
numeric_df = df.select_dtypes(include=[np.number]).copy()

# Feature names used for prediction
feature_names = numeric_df.columns.tolist()

# If model training used different order, adjust here
input_features = feature_names

# ----------------------------------------------------------
# SAFE STAT Fallbacks
# ----------------------------------------------------------
# Replace NaN with safe defaults
col_min = numeric_df.min(skipna=True).fillna(0)
col_max = numeric_df.max(skipna=True).fillna(100)

# ----------------------------------------------------------
# USER INPUT UI
# ----------------------------------------------------------
st.subheader("🧪 Enter Environmental Parameters")

user_input = {}

cols = st.columns(3)

for idx, feat in enumerate(input_features):
    with cols[idx % 3]:
        safe_min = float(col_min[feat]) if np.isfinite(col_min[feat]) else 0.0
        safe_max = float(col_max[feat]) if np.isfinite(col_max[feat]) else safe_min + 100
        
        # Ensure safe bounds
        if safe_min == safe_max:
            safe_max = safe_min + 1

        user_input[feat] = st.slider(
            label=feat.replace("_", " ").title(),
            min_value=float(safe_min),
            max_value=float(safe_max),
            value=float((safe_min + safe_max) / 2),
            step=0.1
        )

# Convert to dataframe for model
input_df = pd.DataFrame([user_input])

# ----------------------------------------------------------
# PREDICT BUTTON
# ----------------------------------------------------------
st.markdown("---")
if st.button("🚀 Predict Fire Risk", use_container_width=True):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"🔥 **High Risk of Fire!** | Probability: {proba:.2%}")
    else:
        st.success(f"🌿 **No Fire Expected** | Probability: {proba:.2%}")

st.markdown("---")
st.info("⚠ This prediction is based on environmental sensor-style input features.")
