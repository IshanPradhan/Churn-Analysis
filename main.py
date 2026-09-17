from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Churn Prediction API")

model = joblib.load('churn_model.pkl')
model_columns = joblib.load('model_columns.pkl')

class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    Tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    numAdminTickets: int
    numTechTickets: int

@app.post("/predict")
def predict(customer: CustomerInput):
    input_df = pd.DataFrame([customer.dict()])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    proba = model.predict_proba(input_df)[0][1]
    prediction = int(proba >= 0.5)

    return {
        "churn_prediction": prediction,
        "churn_probability": round(float(proba), 4)
    }

@app.get("/")
def health_check():
    return {"status": "ok"}