from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_normal():
    # Setup model mock by assuming the app will load the trained model in its lifespan
    # This requires the model to be generated first. The test command should be run after generate_data and train
    payload = {
        "amount": 10.5,
        "time_since_last_tx": 24.0,
        "account_age_days": 365.0,
        "location_risk_score": 0.1
    }
    with TestClient(app) as client_lifespan:
        response = client_lifespan.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert data["is_fraud"] in [0, 1]

def test_predict_invalid_data():
    payload = {
        "amount": "abc", # invalid type
        "time_since_last_tx": 24.0,
        "account_age_days": 365.0,
        "location_risk_score": 0.1
    }
    with TestClient(app) as client_lifespan:
        response = client_lifespan.post("/predict", json=payload)
        assert response.status_code == 422
