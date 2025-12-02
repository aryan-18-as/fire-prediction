import streamlit as st
import numpy as np
import pandas as pd
import onnxruntime as ort
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("🔥 Forest Fire Risk Prediction")

# ---------------------------------------------------
# Load dataset
# ---------------------------------------------------
df = pd.read_csv("Algerian_forest_fires_dataset.csv")
df.columns = df.columns.str.lower().str.strip()

# Auto-detect possible target column (optional)
possible_targets = ["classes", "class", "fire", "label", "target"]
target_col = next((col for col in df.columns if col in possible_targets), None)

# Clean non-numeric columns & force numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

numeric_df = df.select_dtypes(include='number')

# ---------------------------------------------------
# Load ONNX models (Scaler + Classifier)
# ---------------------------------------------------
scaler = ort.InferenceSession("fire_scaler.onnx")
classifier = ort.InferenceSession("fire_classifier.onnx")

# ---------------------------------------------------
# Feature Names
# ---------------------------------------------------
feature_names = [
    "day", "month", "year", "temperature", "rh",
    "ws", "rain", "ffmc", "dmc", "dc", "isi", "bui", "fwi"
]

# ---------------------------------------------------
# UI Form for Inputs
# ---------------------------------------------------
st.subheader("🧪 Enter Environmental Parameters")

user_inputs = []
cols = st.columns(3)

for i, feat in enumerate(feature_names):
    with cols[i % 3]:

        if feat in numeric_df.columns:
            min_val = float(numeric_df[feat].min())
            max_val = float(numeric_df[feat].max())
            default_val = float(numeric_df[feat].mean())
        else:
            min_val, max_val, default_val = 0, 100, 50
        
        value = st.slider(
            label=feat.upper(),
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=0.1
        )
        user_inputs.append(value)

# Convert to array
input_array = np.array([user_inputs], dtype=np.float32)

# ---------------------------------------------------
# Prediction Button
# ---------------------------------------------------
st.markdown("---")

if st.button("🚀 Predict Fire Risk", use_container_width=True):

    # Scale input using ONNX scaler
    scaled = scaler.run(None, {"input": input_array})[0]

    # Predict using ONNX classifier
    pred = classifier.run(None, {"input": scaled})[0]
    predicted_class = int(np.argmax(pred))
    probability = float(np.max(pred))  # best prediction prob

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Fire Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "orange"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Final Classification
    if predicted_class == 1:
        st.error(f"🔥 **High Fire Risk Detected!** (Probability: {probability:.2%})")
    else:
        st.success(f"🌿 **Low Fire Risk (Safe)** (Probability: {probability:.2%})")
