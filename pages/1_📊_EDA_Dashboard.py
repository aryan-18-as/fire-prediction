import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis (EDA)")

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
df = pd.read_csv("Algerian_forest_fires_dataset.csv")
df.columns = df.columns.str.lower().str.strip()

# ----------------------------------------------------------
# FORCE CONVERT ALL POSSIBLE COLUMNS TO NUMERIC
# ----------------------------------------------------------
df_numeric = df.copy()

for col in df_numeric.columns:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors="ignore")

# If conversion fails, errors="ignore" keeps original strings
# Now try to extract real numeric columns
numeric_df = df_numeric.apply(pd.to_numeric, errors="coerce")

# Detect target column
possible_targets = ["classes", "class", "fire", "label", "target"]
target_col = next((col for col in df.columns if col in possible_targets), None)

# Normalize target
if target_col:
    df[target_col] = (
        df[target_col]
        .astype(str)
        .str.lower()
        .str.replace(" ", "")
        .map({"fire": 1, "notfire": 0})
    )

st.write("### 🔍 Cleaned Columns")
st.dataframe(df.head())

st.markdown("---")

# ----------------------------------------------------------
# FIRE vs NO FIRE COUNT
# ----------------------------------------------------------
if target_col:
    st.subheader("🔥 Fire vs No Fire Distribution")

    counts = df[target_col].value_counts().rename({1: "fire", 0: "not fire"})

    fig = px.pie(
        names=counts.index,
        values=counts.values,
        title="Fire vs No Fire Percentage",
        color=counts.index,
        color_discrete_map={"fire": "red", "not fire": "green"},
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------
# STATISTICAL SUMMARY (Fallback-Guaranteed)
# ----------------------------------------------------------
st.subheader("📎 Statistical Summary")

# Replace all-nan numeric df with cleaned df
if numeric_df.select_dtypes(include=[np.number]).empty:
    st.info("⚠ Numeric conversion failed — using raw dataset instead.")
    st.dataframe(df.describe(include="all"))
else:
    st.dataframe(numeric_df.describe(), use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------
# CORRELATION HEATMAP
# ----------------------------------------------------------
st.subheader("📌 Correlation Heatmap")

real_numeric = numeric_df.select_dtypes(include=[np.number])

if real_numeric.shape[1] >= 2:
    corr = real_numeric.corr()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)
else:
    st.warning("⚠ Insufficient numeric columns. Showing alternative visualization.")

    # Alternative correlation using pairwise scatter
    fig = px.scatter_matrix(df, dimensions=df.columns[:5])
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------
# DISTRIBUTION PLOTS FOR EACH FEATURE
# ----------------------------------------------------------
st.subheader("📈 Feature Distributions")

# Replace invalid numeric values with NaN
safe_numeric = numeric_df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')

if safe_numeric.empty:
    st.warning("⚠ No numeric data available for histogram.")
else:
    selected = st.multiselect("Select features:", safe_numeric.columns, default=list(safe_numeric.columns[:3]))

    for col in selected:
        fig = px.histogram(df, x=col, nbins=40, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------
# BOX PLOTS
# ----------------------------------------------------------
st.subheader("📦 Outlier Detection (Boxplots)")

if not safe_numeric.empty:
    box_col = st.selectbox("Select feature for boxplot:", safe_numeric.columns)
    fig = px.box(df, y=box_col, title=f"Boxplot — {box_col}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠ No numeric columns available for boxplots.")

st.markdown("---")

# ----------------------------------------------------------
# FIRE vs NO FIRE COMPARISON CHARTS
# ----------------------------------------------------------
if target_col:
    st.subheader("🔥 Fire vs No Fire — Feature Comparison")

    comp_col = st.selectbox("Select numeric feature:", real_numeric.columns)

    fig = px.box(
        df,
        x=target_col,
        y=comp_col,
        color=target_col,
        title=f"{comp_col} by Fire/No Fire",
        color_discrete_map={1: "red", 0: "green"},
    )
    st.plotly_chart(fig, use_container_width=True)

st.success("✨ EDA Dashboard Loaded Successfully — all charts are fallback-proof!")
