import io
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session
from app.config import FEATURES, MODEL_PATH
from app.database import Base, engine, get_db
from app.models import PredictionRecord
from app.ml import load_model, predict_rows, feature_importance

class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    customer_id: str | None = Field(default=None, max_length=100)
    tenure: int = Field(ge=0, le=120)
    contract: str
    monthly_charges: float = Field(ge=0, le=100000)
    total_charges: float = Field(ge=0, le=10000000)
    internet_service: str
    tech_support: str
    online_security: str
    payment_method: str
    paperless_billing: str
    senior_citizen: int = Field(ge=0, le=1)
    partner: str
    dependents: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    load_model()
    yield

app = FastAPI(title="Customer Churn Predictor API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def save(db, source, result):
    row = PredictionRecord(customer_id=result["customer_id"], churn_prediction=result["churn_prediction"],
        churn_probability=result["churn_probability"], risk_level=result["risk_level"],
        contract=str(source["contract"]), tenure=int(source["tenure"]),
        monthly_charges=float(source["monthly_charges"]))
    db.add(row); db.commit(); db.refresh(row)
    return row

def serialize(r):
    return {"id":r.id,"customer_id":r.customer_id,"churn_prediction":r.churn_prediction,
        "churn_probability":r.churn_probability,"risk_level":r.risk_level,"contract":r.contract,
        "tenure":r.tenure,"monthly_charges":r.monthly_charges,
        "created_at":r.created_at.isoformat() if r.created_at else None}

@app.get("/")
def root():
    return {"message":"Customer Churn Predictor API","docs":"/docs"}

@app.get("/api/health")
def health():
    return {"status":"ok","model_loaded":MODEL_PATH.exists(),"model_type":"RandomForestClassifier"}

@app.get("/api/dashboard")
def dashboard(db: Session = Depends(get_db)):
    _, metrics = load_model()
    rows = db.query(PredictionRecord).order_by(PredictionRecord.id.desc()).all()
    risk_counts = {key:sum(r.risk_level==key for r in rows) for key in ["Low","Medium","High"]}
    total = len(rows)
    return {"summary":{"total_predictions":total,"predicted_churn":sum(r.churn_prediction for r in rows),
        "high_risk":risk_counts["High"],"average_churn_probability":round(sum(r.churn_probability for r in rows)/total,4) if total else 0},
        "risk_distribution":[{"name":k,"value":v} for k,v in risk_counts.items()],
        "model_metrics":metrics,"feature_importance":feature_importance(),
        "recent_predictions":[serialize(r) for r in rows[:8]]}

@app.post("/api/predict")
def predict_one(payload: CustomerInput, db: Session = Depends(get_db)):
    source = payload.model_dump()
    try: result = predict_rows([source])[0]
    except Exception as exc: raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
    row = save(db,source,result)
    return {**result,"prediction_id":row.id}

@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    raw = await file.read()
    if len(raw)>5*1024*1024: raise HTTPException(413,"CSV must be 5 MB or smaller.")
    try: frame = pd.read_csv(io.BytesIO(raw))
    except Exception as exc: raise HTTPException(400,f"Could not read CSV: {exc}") from exc
    missing = set(FEATURES)-set(frame.columns)
    if missing: raise HTTPException(400,f"Missing required columns: {', '.join(sorted(missing))}")
    if frame.empty or len(frame)>5000: raise HTTPException(400,"CSV must contain 1 to 5,000 rows.")
    try:
        sources = frame.to_dict(orient="records")
        results = predict_rows(sources)
        output = []
        for source,result in zip(sources,results):
            row=save(db,source,result)
            output.append({**source,**result,"prediction_id":row.id})
        csv_bytes=pd.DataFrame(output).to_csv(index=False).encode("utf-8")
    except Exception as exc: raise HTTPException(400,f"Could not process CSV: {exc}") from exc
    return StreamingResponse(io.BytesIO(csv_bytes),media_type="text/csv",
        headers={"Content-Disposition":'attachment; filename="churn_predictions.csv"'})

@app.get("/api/customers")
def history(limit:int=100, db:Session=Depends(get_db)):
    rows=db.query(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(min(max(limit,1),500)).all()
    return {"items":[serialize(r) for r in rows],"count":len(rows)}

@app.delete("/api/customers/history")
def clear_history(db:Session=Depends(get_db)):
    db.query(PredictionRecord).delete(); db.commit()
    return {"message":"Prediction history cleared."}
