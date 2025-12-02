import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 EDA Dashboard")

df = pd.read_csv("Algerian_forest_fires_dataset.csv")
df.columns = df.columns.str.lower()

st.subheader("Class Distribution")
fig = px.pie(df, names="classes", title="Fire vs No Fire Distribution")
st.plotly_chart(fig)
