import streamlit as st
import pandas as pd

# ---------------------------------------------------------------
# 🔧 STREAMLIT PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Forest Fire Prediction Dashboard",
    page_icon="🔥",
    layout="wide",
)

# ---------------------------------------------------------------
# 🎨 CUSTOM CSS FOR PREMIUM LOOK
# ---------------------------------------------------------------
st.markdown("""
<style>
/* Main background */
body {
    background-color: #0f1117 !important;
}

/* Section headers */
h1, h2, h3 {
    font-weight: 700 !important;
}

/* Metric styling */
.metric-box {
    background-color: #1f2128;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #2a2d35;
}

/* Summary card */
.summary-card {
    padding: 25px;
    background-color: #1f2128;
    border-radius: 15px;
    border: 1px solid #30333a;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# 📥 LOAD DATASET
# ---------------------------------------------------------------
df = pd.read_csv("Algerian_forest_fires_dataset.csv")
df.columns = df.columns.str.strip().str.lower()  # clean names

# Auto-detect class column
possible_targets = ["classes", "class", "fire", "label", "target"]
target_col = None
for col in df.columns:
    if col in possible_targets:
        target_col = col
        break

# If not found, show error
if target_col is None:
    st.error("❌ No class/target column found in dataset.")
    st.stop()

# Normalize values: convert "fire"/"not fire" to 1/0
df[target_col] = (
    df[target_col]
    .astype(str)
    .str.lower()
    .str.replace(" ", "")
    .map({"fire": 1, "notfire": 0})
)

# ---------------------------------------------------------------
# 🏠 HEADER SECTION
# ---------------------------------------------------------------
st.markdown("<h1>🔥 AI-Based Forest Fire Prediction System</h1>", unsafe_allow_html=True)

st.image(
    "https://images.unsplash.com/photo-1509644851169-2acc08aa25f0",
    use_column_width=True
)

st.markdown("## 📌 Dataset Summary")


# ---------------------------------------------------------------
# 📊 SUMMARY CARDS
# ---------------------------------------------------------------
total_samples = len(df)
fire_cases = int(df[target_col].sum())
no_fire_cases = int(total_samples - fire_cases)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <h3>Total Samples</h3>
        <h1>{total_samples}</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <h3>Fire Cases</h3>
        <h1 style="color:#ff4b4b">{fire_cases}</h1>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <h3>No Fire Cases</h3>
        <h1 style="color:#4bb543">{no_fire_cases}</h1>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------
# 📈 VALUE COUNTS TABLE (Clean UI)
# ---------------------------------------------------------------
st.markdown("### 🔥 Class Distribution Table")

value_counts = df[target_col].value_counts().rename({1: "fire", 0: "not fire"})
value_counts = value_counts.reset_index()
value_counts.columns = ["class", "count"]

st.markdown("""
<div class="summary-card">
""", unsafe_allow_html=True)

st.dataframe(value_counts, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)
