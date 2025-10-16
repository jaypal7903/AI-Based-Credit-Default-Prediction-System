import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and scaler
rf_model = joblib.load("rf_credit_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Card Default Prediction")
st.write("Enter client details to predict the probability of default.")

# Input fields (example for main features, you can add all)
LIMIT_BAL = st.number_input("Credit Limit (LIMIT_BAL)", min_value=0)
SEX = st.selectbox("Gender (1=Male, 2=Female)", [1, 2])
EDUCATION = st.selectbox("EDUCATION (1=Post-Graduate,2=Graduate,3=HighSchool,4=Other)", [1,2,3,4])
MARRIAGE = st.selectbox("MARRIAGE (1=Married,2=UnMarried,3=Divorced)", [1,2,3])
AGE = st.number_input("AGE", min_value=18, max_value=100)

# Last 6 months payment status
st.subheader("Payment Status for Last 6 Months")

status_labels = {
    -2: "No consumption that month (-2)",
    -1: "Paid in full (-1)",
     0: "Paid on time (0)",
     1: "1 month delay (1)",
     2: "2 months delay (2)",
     3: "3 months delay (3)",
     4: "4 months delay (4)",
     5: "5 months delay (5)",
     6: "6 months delay (6)",
     7: "7 months delay (7)",
     8: "8 months delay (8)"
}

PAY_0 = st.selectbox("PAY_0 - Last month payment status", options=list(status_labels.keys()),
                     format_func=lambda x: status_labels[x])
PAY_2 = st.selectbox("PAY_2 - Payment status (2 months ago)", options=list(status_labels.keys()),
                     format_func=lambda x: status_labels[x])
PAY_3 = st.selectbox("PAY_3 - Payment status (3 months ago)", options=list(status_labels.keys()),
                     format_func=lambda x: status_labels[x])
PAY_4 = st.selectbox("PAY_4 - Payment status (4 months ago)", options=list(status_labels.keys()),
                     format_func=lambda x: status_labels[x])
PAY_5 = st.selectbox("PAY_5 - Payment status (5 months ago)", options=list(status_labels.keys()),
                     format_func=lambda x: status_labels[x])
PAY_6 = st.selectbox("PAY_6 - Payment status (6 months ago)", options=list(status_labels.keys()),
                     format_func=lambda x: status_labels[x])


# Bill amounts
st.subheader("Bill Amounts (Past 6 Months)")

BILL_AMT1 = st.number_input("Bill Amount Of Last months ago")
BILL_AMT2 = st.number_input("Bill Amount Of 2 months ago")
BILL_AMT3 = st.number_input("Bill Amount Of 3 months ago")
BILL_AMT4 = st.number_input("Bill Amount Of 4 months ago")
BILL_AMT5 = st.number_input("Bill Amount Of 5 months ago")
BILL_AMT6 = st.number_input("Bill Amount Of 6 months ago")

# Payment amounts
st.subheader("Payment Amounts Made (Past 6 Months)")

PAY_AMT1 = st.number_input("Amount paid last month (most recent bill)", min_value=0.0, step=100.0)
PAY_AMT2 = st.number_input("Amount paid 2 months ago", min_value=0.0, step=100.0)
PAY_AMT3 = st.number_input("Amount paid 3 months ago", min_value=0.0, step=100.0)
PAY_AMT4 = st.number_input("Amount paid 4 months ago", min_value=0.0, step=100.0)
PAY_AMT5 = st.number_input("Amount paid 5 months ago", min_value=0.0, step=100.0)
PAY_AMT6 = st.number_input("Amount paid 6 months ago", min_value=0.0, step=100.0)

# Prepare input dataframe
input_data = pd.DataFrame({
    "LIMIT_BAL":[LIMIT_BAL],
    "SEX":[SEX],
    "EDUCATION":[EDUCATION],
    "MARRIAGE":[MARRIAGE],
    "AGE":[AGE],
    "PAY_0":[PAY_0],
    "PAY_2":[PAY_2],
    "PAY_3":[PAY_3],
    "PAY_4":[PAY_4],
    "PAY_5":[PAY_5],
    "PAY_6":[PAY_6],
    "BILL_AMT1":[BILL_AMT1],
    "BILL_AMT2":[BILL_AMT2],
    "BILL_AMT3":[BILL_AMT3],
    "BILL_AMT4":[BILL_AMT4],
    "BILL_AMT5":[BILL_AMT5],
    "BILL_AMT6":[BILL_AMT6],
    "PAY_AMT1":[PAY_AMT1],
    "PAY_AMT2":[PAY_AMT2],
    "PAY_AMT3":[PAY_AMT3],
    "PAY_AMT4":[PAY_AMT4],
    "PAY_AMT5":[PAY_AMT5],
    "PAY_AMT6":[PAY_AMT6],
})

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Default Probability"):
    prob_default = rf_model.predict_proba(input_scaled)[0][1]
    st.write(f"Predicted Probability of Default: {prob_default:.2f}")
    if prob_default > 0.5:
        st.warning("⚠️ High risk of default!")
    else:
        st.success("✅ Low risk of default.")