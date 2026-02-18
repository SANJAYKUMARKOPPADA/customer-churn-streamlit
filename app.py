import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("📊 Customer Churn Prediction")

st.write("Enter customer details:")

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])

tenure = st.number_input("Tenure (Months)", min_value=0)
monthly = st.number_input("Monthly Charges", min_value=0.0)
total = st.number_input("Total Charges", min_value=0.0)

# Encode inputs
gender = 1 if gender == "Male" else 0
senior = 1 if senior == "Yes" else 0
partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0

# Remaining features (default values)
input_data = [[
    gender, senior, partner, dependents, tenure,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1,
    monthly, total
]]

input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    result = model.predict(input_scaled)
    if result[0] == 1:
        st.error("🔴 Customer WILL CHURN")
    else:
        st.success("🟢 Customer WILL STAY")
