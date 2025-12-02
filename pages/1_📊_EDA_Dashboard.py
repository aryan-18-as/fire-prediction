import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Algerian_forest_fires_dataset.csv")

df = load_data()

st.title("📊 Exploratory Data Analysis (EDA) Dashboard")
st.markdown("This dashboard provides a detailed exploratory analysis of the **Algerian Forest Fires dataset**.")

# ----------------------------------------------------------
# CLEANING (ensure consistent formatting)
# ----------------------------------------------------------
df.columns = df.columns.str.lower().str.strip()

if "classes" in df.columns:
    df["classes"] = df["classes"].astype(str).str.lower().str.strip()

# ----------------------------------------------------------
# SECTION 1 — Dataset Overview
# ----------------------------------------------------------
st.header("📌 Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", df.shape[0])
col2.metric("Total Features", df.shape[1])
col3.metric("Fire Cases", df["classes"].str.contains("fire").sum())

st.dataframe(df.head(), use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------
# SECTION 2 — Class Distribution
# ----------------------------------------------------------
st.subheader("🔥 Class Distribution")

if df["classes"].dtype == "object":
    class_counts = df["classes"].value_counts()
else:
    # fallback if label converted to 0/1
    class_counts = df["classes"].map({0: "not fire", 1: "fire"}).value_counts()

fig = px.pie(
    values=class_counts.values,
    names=class_counts.index,
    title="Fire vs. No Fire Distribution",
    hole=0.4,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------
# SECTION 3 — Numerical Summary
# ----------------------------------------------------------
st.subheader("📎 Statistical Summary (Numerical Columns Only)")

num_df = df.select_dtypes(include=['number'])

if num_df.empty:
    st.warning("⚠ No numeric columns found.")
else:
    st.dataframe(num_df.describe(), use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------
# SECTION 4 — Correlation Heatmap
# ----------------------------------------------------------
st.subheader("📌 Correlation Heatmap")

numeric_df = df.select_dtypes(include=['number'])

if numeric_df.shape[1] < 2:
    st.warning("⚠ Not enough numeric columns to generate correlation heatmap.")
else:
    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar=True,
        ax=ax,
    )
    st.pyplot(fig)

st.markdown("---")

# ----------------------------------------------------------
# SECTION 5 — Feature Distributions
# ----------------------------------------------------------
st.subheader("📈 Feature Distributions")

numeric_cols = numeric_df.columns.tolist()

if numeric_cols:
    selected_cols = st.multiselect(
        "Select features to visualize:",
        numeric_cols,
        default=numeric_cols[:3]
    )

    for col in selected_cols:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No numeric columns to visualize.")

st.markdown("---")

# ----------------------------------------------------------
# SECTION 6 — Boxplots
# ----------------------------------------------------------
st.subheader("📦 Boxplots for Outlier Detection")

if numeric_cols:
    selected_box = st.selectbox("Select feature:", numeric_cols)

    fig = px.box(df, y=selected_box, title=f"Boxplot for {selected_box}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No numeric columns available.")

st.markdown("---")

# ----------------------------------------------------------
# END
# ----------------------------------------------------------
st.success("✨ EDA Dashboard Loaded Successfully!")
