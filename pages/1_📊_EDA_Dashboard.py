import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.title("📊 Exploratory Data Analysis (EDA)")

df = pd.read_csv("Algerian_forest_fires_dataset.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("🔥 Class Distribution")
fig = px.pie(df, names="classes", title="Fire vs No Fire Distribution", hole=0.4)
st.plotly_chart(fig)

st.subheader("📌 Correlation Heatmap")
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), cmap="coolwarm")
st.pyplot(plt)

st.subheader("📈 Monthly Fire Trends")
fig2 = px.histogram(df, x="month", color="classes", barmode="group")
st.plotly_chart(fig2)

st.subheader("🌡 Temperature vs Fire Scatter")
fig3 = px.scatter(df, x="temperature", y="fwi", color="classes")
st.plotly_chart(fig3)
