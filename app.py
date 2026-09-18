import streamlit as st
import pandas as pd
import joblib

model = joblib.load('churn_model.pkl')
expected_columns = joblib.load('model_columns.pkl')

st.title("Churn Prediction App")
st.markdown("Provide the following details ")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
Tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=0)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=100.0, value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=0.0)
numAdminTickets = st.number_input("Number of Admin Tickets", min_value=0, max_value=10, value=0)
numTechTickets = st.number_input("Number of Tech Tickets", min_value=0, max_value=10, value=0)

if st.button("Predict"):
    raw_input = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "Tenure": Tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "numAdminTickets": numAdminTickets,
        "numTechTickets": numTechTickets
    }

    input_df = pd.DataFrame([raw_input])

    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        input_df[col] = input_df[col].map({"Yes": 1, "No": 0})

    multi_cat_cols = ["gender", "MultipleLines", "InternetService", "OnlineSecurity",
                       "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                       "StreamingMovies", "Contract", "PaymentMethod"]
    input_df = pd.get_dummies(input_df, columns=multi_cat_cols)

    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.write(f"Churn Prediction: {'Yes' if prediction == 1 else 'No'}")
    st.write(f"Churn Probability: {probability:.1%}")