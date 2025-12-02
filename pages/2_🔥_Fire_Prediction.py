import streamlit as st
import numpy as np
import pandas as pd
import onnxruntime as ort
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🔥 Forest Fire Risk Prediction")

# Load & prepare dataset
df = pd.read_csv("Algerian_forest_fires_dataset.csv")
df.columns = df.columns.str.strip().str.lower()

# Detect target
possible_targets = ["classes", "class", "fire", "label", "target"]
target_col = next((col for col in df.columns if col in possible_targets), None)

df[target_col] = (
    df[target_col]
    .astype(str)
    .str.lower()
    .str.replace(" ", "")
    .map({"fire": 1, "notfire": 0})
)

# Feature list
features = [
    "day", "month", "year", "temperature", "rh", "ws", "rain",
    "ffmc", "dmc", "dc", "isi", "bui", "fwi"
]

# Load ONNX models
scaler = ort.InferenceSession("fire_scaler.onnx")
classifier = ort.InferenceSession("fire_classifier.onnx")

# UI Layout
left, right = st.columns([2, 1])

inputs = []

with left:
    st.subheader("🧪 Enter Environmental Parameters")

    for feat in features:
        val = st.slider(
            feat,
            min_value=float(df[feat].min()),
            max_value=float(df[feat].max()),
            value=float(df[feat].mean()),
            step=0.1
        )
        inputs.append(val)

# Prediction
if st.button("🔥 Predict Fire Risk", use_container_width=True):
    
    arr = np.array([inputs], dtype=np.float32)

    # Scale Input
    scaled = scaler.run(None, {"input": arr})[0]

    # Predict
    pred = classifier.run(None, {"input": scaled})[0]
    probability = float(np.max(pred))
    result = int(pred.argmax())

    with right:
        st.subheader("📊 Prediction Result")

        # Gauge Chart
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Fire Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, 40], 'color': "green"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"},
                    ]
                }
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Final Verdict
        if result == 1:
            st.error("🔥 **High Risk: Fire may occur**")
        else:
            st.success("🌿 **Low Risk: Safe conditions**")
