# Customer Churn Predictor

Full-stack ML demo using React + Vite, FastAPI, scikit-learn, pandas, and SQLite.

> The built-in dataset is synthetic and is for learning/demo purposes. Evaluation metrics from synthetic data are not real-world performance. Use representative, authorized data before making business decisions.

## Requirements
- Python 3.11+ recommended
- Node.js 20+ recommended

## Run backend (Windows PowerShell)
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m app.train
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
If `py` is unavailable, use `python -m venv .venv`. If PowerShell blocks activation:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

API docs: http://127.0.0.1:8000/docs

## Run frontend (new terminal)
```powershell
cd frontend
npm install
npm run dev
```
Open http://127.0.0.1:5173

## Tests
With backend venv active, run `pytest` inside `backend/`.

## CSV input columns
`tenure, contract, monthly_charges, total_charges, internet_service, tech_support, online_security, payment_method, paperless_billing, senior_citizen, partner, dependents`
Optional: `customer_id`.

## Concepts
- Classification predicts churn/stay.
- A stratified train/test split evaluates unseen rows.
- A scikit-learn Pipeline combines imputation, one-hot encoding, and Random Forest.
- Precision, recall, F1, and ROC-AUC complement accuracy.
- Feature importance describes model behavior, not causation.
