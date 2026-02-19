import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("📊 Customer Churn Prediction")

# EXACT column order used during training
columns = [
    'gender','SeniorCitizen','Partner','Dependents','tenure',
    'PhoneService','MultipleLines','InternetService','OnlineSecurity',
    'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
    'StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
    'MonthlyCharges','TotalCharges'
]

st.subheader("Enter customer details")

gender = st.selectbox("Gender", ["Female","Male"])
senior = st.selectbox("Senior Citizen", ["No","Yes"])
partner = st.selectbox("Partner", ["No","Yes"])
dependents = st.selectbox("Dependents", ["No","Yes"])

tenure = st.number_input("Tenure (months)", 0, 72, 1)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 800.0)

# Encode inputs (same logic as training)
input_dict = {
    'gender': 1 if gender=="Male" else 0,
    'SeniorCitizen': 1 if senior=="Yes" else 0,
    'Partner': 1 if partner=="Yes" else 0,
    'Dependents': 1 if dependents=="Yes" else 0,
    'tenure': tenure,

    # default service values
    'PhoneService': 1,
    'MultipleLines': 1,
    'InternetService': 1,
    'OnlineSecurity': 0,
    'OnlineBackup': 0,
    'DeviceProtection': 0,
    'TechSupport': 0,
    'StreamingTV': 0,
    'StreamingMovies': 0,
    'Contract': 0,              # Month-to-month
    'PaperlessBilling': 1,
    'PaymentMethod': 0,         # Electronic check
    'MonthlyCharges': monthly,
    'TotalCharges': total
}

# Build DataFrame in EXACT training order
input_df = pd.DataFrame([[input_dict[col] for col in columns]], columns=columns)

# Scale + predict
scaled = scaler.transform(input_df)
prob = model.predict_proba(scaled)[0][1]

st.write(f"### Churn Risk: {prob:.2f}")

if prob >= 0.4:
    st.error("🔴 Customer WILL CHURN")
else:
    st.success("🟢 Customer WILL STAY")
