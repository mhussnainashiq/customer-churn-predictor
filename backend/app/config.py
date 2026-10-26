from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "churn_model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"
DB_PATH = DATA / "history.db"
DATA.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)

NUMERIC = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]
CATEGORICAL = ["contract", "internet_service", "tech_support", "online_security",
               "payment_method", "paperless_billing", "partner", "dependents"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churn"
