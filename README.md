# ML2 Assignment – Bike Demand (Tasks 1–3)

This repo contains:
- **Task 1**: Baseline Linear Regression vs Improved Ridge Regression (alpha=0.5), tracked with MLflow (optional).
- **Task 2**: Data drift + performance drift analysis (2011 vs 2012) with an Evidently report.
- **Task 3**: Project packaged into a GitHub-friendly structure (src/, tests/, data/, requirements.txt, README, saved model).

## Folder structure
- `src/` – Core scripts:
  - `train_task1.py` trains and saves the best model to `models/best_model.joblib`
  - `drift_task2.py` generates drift artifacts and evaluates performance on 2011 vs 2012
- `tests/` – Basic test (`pytest`)
- `data/` – `day_2011.csv`, `day_2012.csv`
- `models/` – Saved model artifacts
- `artifacts/` – Generated reports/plots (created after running Task 2)

## Setup
```bash
pip install -r requirements.txt
```

(Optional) Start MLflow UI:
```bash
mlflow server --host 127.0.0.1 --port 5000
```
Then open: http://127.0.0.1:5000

## Run Task 1 (train + save model)
```bash
python src/train_task1.py
```

## Run Task 2 (drift + performance evaluation)
```bash
python src/drift_task2.py
```

## Run tests
```bash
pytest
```
