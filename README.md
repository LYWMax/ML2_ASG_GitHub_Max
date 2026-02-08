# Bike Demand Prediction & Data Drift Monitoring

## Project Overview
This project implements a machine learning pipeline to predict daily bike rental demand and monitor model reliability over time. The system includes model training, performance evaluation, data drift detection, and automated testing using GitHub Actions.

The model was trained using the Bike Sharing Dataset (2011–2012). A regression model was developed using scikit-learn and evaluated using RMSE, MAE, and R² metrics. Data drift between the 2011 and 2012 datasets was analysed to study its impact on model performance.

This repository demonstrates a basic MLOps workflow including:
- Model development
- Model evaluation
- Data drift monitoring
- Automated testing (Quality Gate)
- Continuous Integration using GitHub Actions

---

## Repository Structure
```
├── src/                # Training and drift analysis scripts
│   ├── train_task1.py
│   ├── drift_task2.py
│   └── utils.py
│
├── data/               # Datasets
│   ├── day_2011.csv
│   └── day_2012.csv
│
├── models/             # Saved trained model
│   └── model.joblib
│
├── tests/              # Automated tests
│   └── test_model.py
│
├── requirements.txt    # Python dependencies
├── README.md
└── .github/workflows/python-app.yml  # CI workflow
```

---

## Requirements
Install all required dependencies using:

```bash
pip install -r requirements.txt
```

Key libraries used:
- pandas
- numpy
- scikit-learn
- mlflow
- evidently
- matplotlib
- seaborn
- pytest

---

## How to Run the Project

### 1. Train the Model (Task 1)
Train the regression model using the 2011 dataset and save the best model:

```bash
python src/train_task1.py
```

Output:
- Saves trained model to `models/model.joblib`
- Displays evaluation metrics

---

### 2. Run Data Drift Analysis (Task 2)
Evaluate model performance on 2011 and 2012 data and generate drift reports:

```bash
python src/drift_task2.py
```

Output:
- Performance comparison (2011 vs 2012)
- Drift statistics
- Evidently drift report (HTML)
- Humidity distribution plot

---

### 3. Run Automated Tests (Quality Gate)
Check whether the trained model meets the required performance threshold:

```bash
pytest
```

The test will:
- Load the saved model
- Generate predictions
- Compute RMSE
- Fail if model performance is worse than the baseline

---

## Continuous Integration (GitHub Actions)
GitHub Actions automatically runs when code is pushed to the `main` branch.

The workflow:
1. Installs dependencies
2. Executes `pytest`
3. Validates model performance

If the test fails, the workflow is marked as failed, preventing acceptance of degraded models.

---

## Model Description
The selected model is a **Ridge Regression** model trained on the 2011 dataset.  
Categorical variables are encoded using One-Hot Encoding, while numerical features are passed directly to the model.

Performance metrics used:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

---

## Data Drift Monitoring
Data drift was analysed by comparing the 2011 (reference) and 2012 (current) datasets. Changes in feature distributions were measured using descriptive statistics and an Evidently data drift report.

Humidity was identified as the primary drifting feature, which contributed to increased prediction error when the model was applied to the 2012 dataset.

---

## Reproducibility
To reproduce the project:

```bash
git clone [<your-repository-url>]
cd <repository-folder>
pip install -r requirements.txt
python src/train_task1.py
python src/drift_task2.py
```

---

## Author
Machine Learning 2 Assignment  
Bike Demand Prediction with Model Monitoring
by Low Yu Wen Max 
