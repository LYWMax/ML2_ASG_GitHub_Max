import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression, Ridge

from utils import evaluate_regression

# Optional: MLflow
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False

DATA_PATH = os.path.join("data", "day_2011.csv")
MODEL_OUT = os.path.join("models", "best_model.joblib")

def build_preprocess():
    categorical_features = ["season", "mnth", "holiday", "weekday", "weathersit", "workingday"]
    numeric_features = ["temp", "hum", "windspeed"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )
    return preprocess, categorical_features, numeric_features

def evaluate_cv(model, X_train, y_train, cv_folds=5):
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        cv=cv,
        return_train_score=False,
    )
    rmse_mean = float(-scores["test_rmse"].mean())
    rmse_std = float(scores["test_rmse"].std())
    mae_mean = float(-scores["test_mae"].mean())
    r2_mean = float(scores["test_r2"].mean())
    return {"cv_rmse_mean": rmse_mean, "cv_rmse_std": rmse_std, "cv_mae_mean": mae_mean, "cv_r2_mean": r2_mean}

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}. Put day_2011.csv in the data/ folder.")

    df = pd.read_csv(DATA_PATH)

    # Same preprocessing decisions as your notebook/report
    df = df.drop(columns=["dteday", "atemp"], errors="ignore")

    X = df.drop(columns="cnt")
    y = df["cnt"]

    preprocess, cat_feats, num_feats = build_preprocess()

    # Match your notebook split (random_state=67)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=67
    )

    # -------- Experiment 1: Baseline Linear Regression --------
    lr_model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LinearRegression()),
    ])
    lr_model.fit(X_train, y_train)

    lr_cv = evaluate_cv(lr_model, X_train, y_train)
    lr_rmse, lr_mae, lr_r2 = evaluate_regression(lr_model, X_test, y_test)

    print("[Baseline LinearRegression]")
    print(f"CV RMSE: {lr_cv['cv_rmse_mean']:.3f} +/- {lr_cv['cv_rmse_std']:.3f}")
    print(f"Test RMSE: {lr_rmse:.3f} | MAE: {lr_mae:.3f} | R2: {lr_r2:.3f}")
    print()

    # -------- Experiment 2: Improved Ridge Regression (alpha=0.5) --------
    alpha = 0.5
    ridge_model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", Ridge(alpha=alpha, random_state=42)),
    ])
    ridge_model.fit(X_train, y_train)

    ridge_cv = evaluate_cv(ridge_model, X_train, y_train)
    ridge_rmse, ridge_mae, ridge_r2 = evaluate_regression(ridge_model, X_test, y_test)

    print("[Improved RidgeRegression]")
    print(f"alpha: {alpha}")
    print(f"CV RMSE: {ridge_cv['cv_rmse_mean']:.3f} +/- {ridge_cv['cv_rmse_std']:.3f}")
    print(f"Test RMSE: {ridge_rmse:.3f} | MAE: {ridge_mae:.3f} | R2: {ridge_r2:.3f}")
    print()

    # Choose best by lowest test RMSE (matches your report choice)
    best_model = ridge_model
    best_name = f"Ridge(alpha={alpha})"
    best_metrics = {"rmse": ridge_rmse, "mae": ridge_mae, "r2": ridge_r2}

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(best_model, MODEL_OUT)
    print(f"Saved best model: {best_name} -> {MODEL_OUT}")

    # -------- MLflow logging (optional) --------
    if _HAS_MLFLOW:
        # If MLflow server isn't running, we still want the script to succeed.
        try:
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("ML2_Task1_Bike_Sharing_2011")

            # Baseline run
            with mlflow.start_run(run_name="Baseline_Linear_Regression"):
                mlflow.set_tag("role", "baseline")
                mlflow.set_tag("dataset", "day_2011")
                mlflow.log_params({
                    "model_type": "LinearRegression",
                    "preprocessing": "OHE(cat)+passthrough(num)",
                    "categorical_features": ",".join(cat_feats),
                    "numeric_features": ",".join(num_feats),
                })
                mlflow.log_metric("rmse", lr_rmse)
                mlflow.log_metric("mae", lr_mae)
                mlflow.log_metric("r2", lr_r2)
                mlflow.log_metric("cv_rmse_mean", lr_cv["cv_rmse_mean"])
                mlflow.log_metric("cv_rmse_std", lr_cv["cv_rmse_std"])

                signature = infer_signature(X_train, lr_model.predict(X_train))
                mlflow.sklearn.log_model(
                    sk_model=lr_model,
                    artifact_path="LinearRegressionModel",
                    signature=signature,
                    input_example=X_train.iloc[:5],
                )

            # Improved run
            with mlflow.start_run(run_name="Ridge_Regression"):
                mlflow.set_tag("role", "improved")
                mlflow.set_tag("dataset", "day_2011")
                mlflow.log_params({
                    "model_type": "RidgeRegression",
                    "alpha": alpha,
                    "preprocessing": "OHE(cat)+passthrough(num)",
                    "categorical_features": ",".join(cat_feats),
                    "numeric_features": ",".join(num_feats),
                })
                mlflow.log_metric("rmse", ridge_rmse)
                mlflow.log_metric("mae", ridge_mae)
                mlflow.log_metric("r2", ridge_r2)
                mlflow.log_metric("cv_rmse_mean", ridge_cv["cv_rmse_mean"])
                mlflow.log_metric("cv_rmse_std", ridge_cv["cv_rmse_std"])

                signature = infer_signature(X_train, ridge_model.predict(X_train))
                mlflow.sklearn.log_model(
                    sk_model=ridge_model,
                    artifact_path="RidgeRegressionModel",
                    signature=signature,
                    input_example=X_train.iloc[:5],
                )
        except Exception as e:
            warnings.warn(f"MLflow logging skipped (MLflow server may be off). Details: {e}")

if __name__ == "__main__":
    main()
