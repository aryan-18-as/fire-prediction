import streamlit as st
import numpy as np
import onnxruntime as ort
import plotly.graph_objects as go

st.title("🔥 Forest Fire Risk Prediction")

st.markdown("""
Adjust the environmental parameters below and click **Predict**  
to estimate whether there is a **risk of forest fire**.
""")

# ---------------------------------------------------
# Load ONNX models (Scaler + Classifier)
# ---------------------------------------------------
@st.cache_resource
def load_sessions():
    scaler_sess = ort.InferenceSession("fire_scaler.onnx")
    clf_sess = ort.InferenceSession("fire_classifier.onnx")
    return scaler_sess, clf_sess

scaler_session, clf_session = load_sessions()

# ---------------------------------------------------
# Feature Names (order MUST match training)
# ---------------------------------------------------
feature_names = [
    "day", "month", "year",
    "temperature", "rh", "ws", "rain",
    "ffmc", "dmc", "dc", "isi", "bui", "fwi"
]

# ---------------------------------------------------
# Hard-coded safe ranges for each feature
# (No dependency on dataset → no slider errors)
# ---------------------------------------------------
ranges = {
    "day":          (1, 31, 15),
    "month":        (1, 12, 6),
    "year":         (2012, 2030, 2012),
    "temperature":  (0.0, 50.0, 30.0),
    "rh":           (0.0, 100.0, 50.0),
    "ws":           (0.0, 40.0, 10.0),
    "rain":         (0.0, 20.0, 0.0),
    "ffmc":         (0.0, 100.0, 80.0),
    "dmc":          (0.0, 300.0, 50.0),
    "dc":           (0.0, 1000.0, 200.0),
    "isi":          (0.0, 50.0, 10.0),
    "bui":          (0.0, 200.0, 40.0),
    "fwi":          (0.0, 50.0, 15.0),
}

# ---------------------------------------------------
# UI – Sliders for all inputs
# ---------------------------------------------------
st.subheader("🧪 Enter Environmental Parameters")

cols = st.columns(3)
user_values = []

for i, feat in enumerate(feature_names):
    min_v, max_v, default_v = ranges[feat]

    # Decide if feature is int or float
    is_int = isinstance(min_v, int) and isinstance(max_v, int)

    with cols[i % 3]:
        if is_int:
            val = st.slider(
                label=feat.upper(),
                min_value=int(min_v),
                max_value=int(max_v),
                value=int(default_v),
                step=1
            )
        else:
            val = st.slider(
                label=feat.upper(),
                min_value=float(min_v),
                max_value=float(max_v),
                value=float(default_v),
                step=0.1
            )
        user_values.append(val)

# Convert to numpy array for ONNX
input_array = np.array([user_values], dtype=np.float32)

st.markdown("---")

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("🚀 Predict Fire Risk", use_container_width=True):
    # Scale input
    scaled = scaler_session.run(None, {"input": input_array})[0]

    # Predict
    logits = clf_session.run(None, {"input": scaled})[0]
    predicted_class = int(np.argmax(logits))
    probability = float(np.max(logits))  # max prob

    # Gauge for probability
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
                {'range': [70, 100], 'color': "red"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Text verdict
    if predicted_class == 1:
        st.error(f"🔥 **High Fire Risk Detected!** (Probability: {probability:.2%})")
    else:
        st.success(f"🌿 **Low Fire Risk (Safe)** (Probability: {probability:.2%})")
