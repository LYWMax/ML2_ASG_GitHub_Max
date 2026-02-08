import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "best_model.joblib")
DATA_PATH = os.path.join("data", "day_2011.csv")

def test_model_predicts_without_error():
    assert os.path.exists(MODEL_PATH), "Missing models/best_model.joblib. Run src/train_task1.py first."
    assert os.path.exists(DATA_PATH), "Missing data/day_2011.csv."

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH).drop(columns=["dteday", "atemp"], errors="ignore")
    X = df.drop(columns="cnt")

    preds = model.predict(X.head(5))
    assert len(preds) == 5
