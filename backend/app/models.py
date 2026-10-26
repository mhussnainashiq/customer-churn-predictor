from datetime import datetime
from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base

class PredictionRecord(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    customer_id: Mapped[str] = mapped_column(String(100), default="Manual entry")
    churn_prediction: Mapped[bool] = mapped_column(Boolean)
    churn_probability: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(20))
    contract: Mapped[str] = mapped_column(String(50))
    tenure: Mapped[int] = mapped_column(Integer)
    monthly_charges: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
