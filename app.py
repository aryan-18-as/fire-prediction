import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Forest Fire Prediction System",
    page_icon="🔥",
    layout="wide",
)

st.title("🔥 AI-Based Forest Fire Prediction System")
st.markdown("""
### Welcome to the intelligent Forest Fire Detection and Prediction System  
This dashboard helps in:
- 🌲 Monitoring forest conditions  
- 🔥 Predicting potential forest fires  
- 📊 Exploring environmental data  
- 🧠 Understanding ML model performance  

Navigate using the sidebar to explore pages.
""")

st.image("https://i.imgur.com/QkQJtCE.jpg", use_column_width=True)

st.markdown("---")
st.subheader("📌 Dataset Summary")

df = pd.read_csv("Algerian_foresr_fires_dataset.csv")
col1, col2, col3 = st.columns(3)

col1.metric("Total Samples", len(df))
col2.metric("Fire Cases", df["classes"].sum())
col3.metric("No-Fire Cases", len(df) - df["classes"].sum())

st.warning("Use the left-side menu to navigate through EDA, Prediction, Performance, and more.")
