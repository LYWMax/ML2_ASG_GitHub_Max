import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

MODEL_PATH = os.path.join("models", "model.joblib")
DATA_PATH = os.path.join("data", "day_2011.csv")

# Set a performance threshold (adjust if needed)
RMSE_THRESHOLD = 700

def test_model_performance():
    # Check files exist
    assert os.path.exists(MODEL_PATH), "Missing models/model.joblib. Run src/train_task1.py first."
    assert os.path.exists(DATA_PATH), "Missing data/day_2011.csv."

    # Load model
    model = joblib.load(MODEL_PATH)

    # Load data
    df = pd.read_csv(DATA_PATH).drop(columns=["dteday", "atemp"], errors="ignore")

    X = df.drop(columns="cnt")
    y = df["cnt"]

    # Predict
    preds = model.predict(X)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y, preds))

    print("RMSE:", rmse)

    # QUALITY GATE
    assert rmse <= RMSE_THRESHOLD, f"Model RMSE {rmse:.2f} exceeds threshold {RMSE_THRESHOLD}"
