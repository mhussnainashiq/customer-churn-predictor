from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def sample():
    return {"customer_id":"TEST-001","tenure":4,"contract":"Month-to-month",
      "monthly_charges":95.5,"total_charges":382,"internet_service":"Fiber optic",
      "tech_support":"No","online_security":"No","payment_method":"Electronic check",
      "paperless_billing":"Yes","senior_citizen":0,"partner":"No","dependents":"No"}

def test_health():
    response=client.get("/api/health")
    assert response.status_code==200 and response.json()["status"]=="ok"

def test_dashboard():
    response=client.get("/api/dashboard")
    assert response.status_code==200 and "model_metrics" in response.json()

def test_predict():
    response=client.post("/api/predict",json=sample())
    assert response.status_code==200, response.text
    data=response.json()
    assert 0 <= data["churn_probability"] <= 1
    assert data["risk_level"] in ["Low","Medium","High"]
