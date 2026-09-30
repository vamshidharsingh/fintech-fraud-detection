from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from contextlib import asynccontextmanager

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load("models/xgboost_model.joblib")
    yield
    model = None

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

class Transaction(BaseModel):
    amount: float
    time_since_last_tx: float
    account_age_days: float
    location_risk_score: float

@app.post("/predict")
def predict_fraud(tx: Transaction):
    data = pd.DataFrame([tx.dict()])
    prob = model.predict_proba(data)[0, 1]
    prediction = int(model.predict(data)[0])
    return {
        "fraud_probability": float(prob),
        "is_fraud": prediction
    }

@app.get("/health")
def health():
    return {"status": "ok"}
