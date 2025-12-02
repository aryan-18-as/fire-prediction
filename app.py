import streamlit as st
import pandas as pd

st.set_page_config(page_title="Forest Fire Prediction", page_icon="🔥", layout="wide")

st.title("🔥 AI-Based Forest Fire Prediction System")

# -----------------------------------------
# 📌 Load and clean dataset safely
# -----------------------------------------
df = pd.read_csv("Algerian_forest_fires_dataset.csv")

# Clean column names (remove spaces, lowercase)
df.columns = df.columns.str.strip().str.lower()

# Auto-detect class column
possible_targets = ["classes", "class", "target", "fire", "label"]

target_col = None
for col in df.columns:
    if col in possible_targets:
        target_col = col
        break

if target_col is None:
    st.error("❌ No valid class/target column found in dataset.")
    st.write("Columns found:", df.columns.tolist())
    st.stop()

# -----------------------------------------
# 📌 Dataset Summary
# -----------------------------------------
st.markdown("---")
st.header("📌 Dataset Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Total Samples", len(df))

# Ensure binary class exists
try:
    col2.metric("Fire Cases", int(df[target_col].sum()))
    col3.metric("No Fire Cases", int((1 - df[target_col]).sum()))
except:
    st.warning("⚠ Target column is not numeric. Showing value counts instead.")
    counts = df[target_col].value_counts()
    st.write(counts)

# -----------------------------------------
# 🖼 Working Forest Fire Banner Image
# -----------------------------------------
st.image("https://images.unsplash.com/photo-1509644851169-2acc08aa25f0",
         use_column_width=True)
