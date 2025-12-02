import streamlit as st

st.title("📘 Project Documentation")

st.markdown("""
## 🔥 AI-Based Forest Fire Prediction System

### **Problem Statement**
Forest fires cause ecological damage and property loss. The goal is to use AI to predict fire risk using environmental variables.

### **Dataset Used**
- Algerian Forest Fire Dataset  
- Combined both Bejaia & Sidi Bel-Abbes regions  
- 244 total records  

### **Algorithms Used**
- Random Forest (Best)
- SVM
- Logistic Regression
- Decision Tree
- XGBoost

### **Evaluation Metrics**
- Accuracy  
- F1 Score  
- Precision  
- Recall  
- ROC-AUC  

### **Deployment**
- Streamlit ONNX-based multi-page dashboard  
- Python 3.13 compatible  
""")
