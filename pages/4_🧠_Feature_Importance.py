import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🧠 Feature Importance")

importance = {
    "feature": ["ffmc","dmc","isi","fwi","temperature","rh","ws","bui","dc","month","day","rain","year"],
    "importance": [0.21,0.18,0.13,0.11,0.09,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.01]
}

df_imp = pd.DataFrame(importance)

fig = px.bar(df_imp, x="importance", y="feature", orientation="h",
             title="Feature Importance (Random Forest)")
st.plotly_chart(fig)
