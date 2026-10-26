import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from app.config import ARTIFACTS, MODEL_PATH, METRICS_PATH, FEATURES, NUMERIC, CATEGORICAL, TARGET, DATA

def demo_data(n=2400, seed=42):
    """Generate reproducible synthetic customer records for demo use only."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.56,.24,.20])
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[.48,.36,.16])
    tech = rng.choice(["Yes", "No", "No internet service"], n, p=[.27,.57,.16])
    security = rng.choice(["Yes", "No", "No internet service"], n, p=[.30,.54,.16])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)
    monthly = np.clip(rng.normal(72, 25, n), 18, 150).round(2)
    total = (monthly * tenure + rng.normal(0, 90, n)).clip(0).round(2)
    senior = rng.choice([0,1], n, p=[.84,.16])
    paperless = rng.choice(["Yes","No"], n, p=[.72,.28])
    partner = rng.choice(["Yes","No"], n)
    dependents = rng.choice(["Yes","No"], n, p=[.30,.70])
    logit = (-1.25 + (contract=="Month-to-month")*1.15 + (contract=="One year")*(-.35)
             + (contract=="Two year")*(-.95) + (tenure<8)*.90 + (tenure>36)*(-.75)
             + (payment=="Electronic check")*.48 + (tech=="No")*.34 + (security=="No")*.27
             + (monthly>95)*.30 + senior*.22 + (partner=="Yes")*(-.18)
             + (dependents=="Yes")*(-.25) + rng.normal(0,.55,n))
    p = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, p)
    return pd.DataFrame({
        "customer_id":[f"DEMO-{i+1:05d}" for i in range(n)], "tenure":tenure,
        "contract":contract, "monthly_charges":monthly, "total_charges":total,
        "internet_service":internet, "tech_support":tech, "online_security":security,
        "payment_method":payment, "paperless_billing":paperless, "senior_citizen":senior,
        "partner":partner, "dependents":dependents, "churn":churn
    })

def training_data():
    path = DATA / "customers.csv"
    df = pd.read_csv(path) if path.exists() else demo_data()
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Training CSV missing columns: {', '.join(sorted(missing))}")
    if not path.exists():
        df.to_csv(DATA / "generated_customers.csv", index=False)
    return df.dropna(subset=[TARGET])

def train():
    df = training_data()
    X, y = df[FEATURES], df[TARGET].astype(int)
    if y.nunique() != 2:
        raise ValueError("Training data must include both churn classes 0 and 1.")
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    category_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("numeric", numeric_pipe, NUMERIC),
                                      ("categorical", category_pipe, CATEGORICAL)])
    model = Pipeline([("preprocessor", preprocessor),
                      ("classifier", RandomForestClassifier(n_estimators=220, min_samples_leaf=3,
                       class_weight="balanced", random_state=42, n_jobs=-1))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.22, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test,pred)),4),
        "precision": round(float(precision_score(y_test,pred,zero_division=0)),4),
        "recall": round(float(recall_score(y_test,pred,zero_division=0)),4),
        "f1_score": round(float(f1_score(y_test,pred,zero_division=0)),4),
        "roc_auc": round(float(roc_auc_score(y_test,probs)),4),
        "confusion_matrix": confusion_matrix(y_test,pred,labels=[0,1]).tolist(),
        "training_rows":len(y_train), "test_rows":len(y_test),
        "dataset_note":"Synthetic demonstration data" if not (DATA/"customers.csv").exists() else "Configured training CSV"
    }
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    return model, metrics

def load_model():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        return train()
    return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text(encoding="utf-8"))

def risk(prob):
    return "High" if prob >= .65 else "Medium" if prob >= .35 else "Low"

def advice(prob, contract, tenure):
    if prob >= .65 and contract == "Month-to-month":
        return "Consider a relevant longer-term plan and a proactive retention conversation."
    if prob >= .65 and tenure < 6:
        return "Offer onboarding help and check whether the service meets expectations."
    if prob >= .65:
        return "Review recent support issues and consider a personalized retention offer."
    if prob >= .35:
        return "Monitor satisfaction and review support needs at the next customer touchpoint."
    return "Maintain service quality and continue normal customer engagement."

def predict_rows(rows):
    model, _ = load_model()
    frame = pd.DataFrame(rows)
    probabilities = model.predict_proba(frame[FEATURES])[:,1]
    labels = model.predict(frame[FEATURES])
    results = []
    for row, label, probability in zip(rows, labels, probabilities):
        p = float(probability)
        results.append({"customer_id":str(row.get("customer_id") or "Manual entry"),
            "churn_prediction":bool(label),"churn_probability":round(p,4),"risk_level":risk(p),
            "recommendation":advice(p,str(row["contract"]),int(row["tenure"]))})
    return results

def feature_importance():
    model, _ = load_model()
    names = model.named_steps["preprocessor"].get_feature_names_out()
    values = model.named_steps["classifier"].feature_importances_
    grouped = {}
    for name, value in zip(names, values):
        name = name.replace("numeric__","").replace("categorical__","")
        feature = name.split("_")[0]
        grouped[feature] = grouped.get(feature,0) + float(value)
    return sorted([{"feature":k,"importance":round(v,5)} for k,v in grouped.items()],
                  key=lambda x:x["importance"],reverse=True)[:10]
