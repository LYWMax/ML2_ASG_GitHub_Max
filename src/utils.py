import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_regression(model, X, y):
    """Return (rmse, mae, r2) for a fitted regression model."""
    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    mae = float(mean_absolute_error(y, preds))
    r2 = float(r2_score(y, preds))
    return rmse, mae, r2
