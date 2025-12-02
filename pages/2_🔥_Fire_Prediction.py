import streamlit as st
import numpy as np
import onnxruntime as ort
import pandas as pd

st.title("🔥 Forest Fire Prediction")

feature_names = [
    "day","month","year","temperature","rh","ws","rain",
    "ffmc","dmc","dc","isi","bui","fwi"
]

df = pd.read_csv("dataset.csv")
min_vals = df[feature_names].min()
max_vals = df[feature_names].max()

# Load ONNX models
scaler_session = ort.InferenceSession("fire_scaler.onnx")
clf_session = ort.InferenceSession("fire_classifier.onnx")

inputs = []
cols = st.columns(3)

for idx, feat in enumerate(feature_names):
    with cols[idx % 3]:
        val = st.number_input(feat, float(min_vals[feat]), float(max_vals[feat]), float(df[feat].mean()))
        inputs.append(val)

if st.button("Predict Fire 🔥"):
    arr = np.array([inputs], dtype=np.float32)

    scaled = scaler_session.run(None, {"input": arr})[0]
    pred = clf_session.run(None, {"input": scaled})[0]
    result = int(pred.argmax())

    if result == 1:
        st.error("🔥 FIRE RISK DETECTED!")
    else:
        st.success("🌿 No Fire Risk Detected")
