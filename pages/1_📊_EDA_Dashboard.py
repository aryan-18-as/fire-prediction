import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("📊 Exploratory Data Analysis (EDA)")

# Load & Clean
df = pd.read_csv("Algerian_forest_fires_dataset.csv")
df.columns = df.columns.str.strip().str.lower()

# Auto-detect target column
possible_targets = ["classes", "class", "fire", "label", "target"]
target_col = None
for col in df.columns:
    if col in possible_targets:
        target_col = col
        break

if target_col is None:
    st.error("❌ No class/target column found.")
    st.stop()

# Normalize fire/not fire
df[target_col] = (
    df[target_col]
    .astype(str)
    .str.lower()
    .str.replace(" ", "")
    .map({"fire": 1, "notfire": 0})
)

# -------------------------------
# 1️⃣ Basic Stats
# -------------------------------
st.subheader("📌 Dataset Overview")
st.dataframe(df.head(), use_container_width=True)

# -------------------------------
# 2️⃣ Class Distribution
# -------------------------------
st.subheader("🔥 Class Distribution")

fire_count = df[target_col].sum()
nofire_count = len(df) - fire_count

pie_fig = px.pie(
    names=["Fire", "No Fire"], 
    values=[fire_count, nofire_count],
    color=["Fire", "No Fire"],
    color_discrete_map={"Fire": "red", "No Fire": "green"},
    hole=0.35,
    title="Fire vs No Fire Percentage"
)

st.plotly_chart(pie_fig, use_container_width=True)

# -------------------------------
# 3️⃣ Correlation Heatmap
# -------------------------------
st.subheader("📌 Correlation Heatmap")

plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
st.pyplot(plt)

# -------------------------------
# 4️⃣ Feature Distribution
# -------------------------------
st.subheader("📈 Feature Distributions")

num_cols = df.drop(columns=[target_col]).columns
selected = st.multiselect("Select Features:", num_cols, default=num_cols[:4])

if selected:
    fig = px.histogram(
        df,
        x=selected,
        nbins=30,
        marginal="box",
        title="Feature Distribution",
        color=target_col,
        color_discrete_map={1: "red", 0: "green"},
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 5️⃣ Scatter Matrix (small sample)
# -------------------------------
st.subheader("🔎 Feature Relationships — Scatter Matrix")

sample_df = df.sample(min(200, len(df)))

fig_matrix = px.scatter_matrix(
    sample_df,
    dimensions=num_cols[:5],
    color=target_col,
    color_discrete_map={1: "red", 0: "green"},
    title="Scatter Matrix (Sampled)",
)

st.plotly_chart(fig_matrix, use_container_width=True)
