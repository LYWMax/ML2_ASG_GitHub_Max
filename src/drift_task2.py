import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils import evaluate_regression

# Optional: MLflow
try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False

DATA_2011 = os.path.join("data", "day_2011.csv")
DATA_2012 = os.path.join("data", "day_2012.csv")
MODEL_PATH = os.path.join("models", "best_model.joblib")

ARTIFACT_DIR = "artifacts"

def ensure_artifacts_dir():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

def main():
    for p in [DATA_2011, DATA_2012]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing dataset: {p}. Put the csv files in the data/ folder.")

    df_2011 = pd.read_csv(DATA_2011).drop(columns=["dteday", "atemp"], errors="ignore")
    df_2012 = pd.read_csv(DATA_2012).drop(columns=["dteday", "atemp"], errors="ignore")

    reference = df_2011.copy()
    current = df_2012.copy()

    X_ref = reference.drop(columns="cnt")
    y_ref = reference["cnt"]
    X_curr = current.drop(columns="cnt")
    y_curr = current["cnt"]

    # Load selected model (Task 1 best model)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing saved model: {MODEL_PATH}. Run src/train_task1.py first.")
    model = joblib.load(MODEL_PATH)

    # -------- Descriptive statistics + drift ratio --------
    summary = pd.DataFrame({
        "mean_2011": X_ref.mean(),
        "mean_2012": X_curr.mean(),
        "std_2011": X_ref.std(),
        "std_2012": X_curr.std(),
    })
    summary["mean_change_abs"] = (summary["mean_2012"] - summary["mean_2011"]).abs()
    summary["std_change_abs"] = (summary["std_2012"] - summary["std_2011"]).abs()
    summary["drift_ratio"] = summary["mean_change_abs"] / summary["std_2011"].replace(0, np.nan)

    summary_sorted = summary.sort_values("drift_ratio", ascending=False)

    ensure_artifacts_dir()
    summary_csv = os.path.join(ARTIFACT_DIR, "drift_summary_stats.csv")
    summary_sorted.to_csv(summary_csv)
    print("Saved drift summary:", summary_csv)

    # -------- Visualisation: humidity boxplot --------
    hum_plot = pd.DataFrame({
        "Humidity": pd.concat([X_ref["hum"], X_curr["hum"]], ignore_index=True),
        "Year": (["2011"] * len(X_ref)) + (["2012"] * len(X_curr)),
    })
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="Year", y="Humidity", data=hum_plot)
    plt.title("Humidity Distribution (2011 vs 2012)")
    plt.xlabel("Year")
    plt.ylabel("Normalized Humidity")
    hum_plot_path = os.path.join(ARTIFACT_DIR, "humidity_boxplot.png")
    plt.tight_layout()
    plt.savefig(hum_plot_path, dpi=200)
    plt.close()
    print("Saved plot:", hum_plot_path)

    # -------- Evidently drift report --------
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=X_ref, current_data=X_curr)

    drift_html_path = os.path.join(ARTIFACT_DIR, "evidently_data_drift_report.html")
    drift_json_path = os.path.join(ARTIFACT_DIR, "evidently_data_drift_report.json")
    drift_report.save_html(drift_html_path)

    report_json = drift_report.as_dict()
    with open(drift_json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    print("Saved Evidently report:", drift_html_path)
    print("Saved Evidently JSON:", drift_json_path)

    # -------- Performance drift (Task 2 requirement) --------
    y_pred_2011 = model.predict(X_ref)
    rmse_2011 = float(np.sqrt(mean_squared_error(y_ref, y_pred_2011)))
    mae_2011 = float(mean_absolute_error(y_ref, y_pred_2011))

    y_pred_2012 = model.predict(X_curr)
    rmse_2012 = float(np.sqrt(mean_squared_error(y_curr, y_pred_2012)))
    mae_2012 = float(mean_absolute_error(y_curr, y_pred_2012))

    print(f"Model performance on 2011 -> RMSE: {rmse_2011:.3f}, MAE: {mae_2011:.3f}")
    print(f"Model performance on 2012 -> RMSE: {rmse_2012:.3f}, MAE: {mae_2012:.3f}")

    # -------- Drift response: retrain on 2011+2012 and compare RMSE on 2012 --------
    df_combined = pd.concat([df_2011, df_2012], ignore_index=True)
    X = df_combined.drop(columns="cnt")
    y = df_combined["cnt"]

    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
        X, y, test_size=0.2, random_state=142
    )

    # Reuse preprocessing & alpha=0.5
    # NOTE: The preprocess is inside the saved pipeline; simplest is to rebuild a fresh Pipeline with same config.
    # Here, we load the model and re-fit it on the new training data.
    retrained_model = joblib.load(MODEL_PATH)
    retrained_model.fit(X_train_new, y_train_new)

    old_rmse_2012 = rmse_2012
    new_pred_2012 = retrained_model.predict(X_curr)
    new_rmse_2012 = float(np.sqrt(mean_squared_error(y_curr, new_pred_2012)))

    print(f"Old model RMSE on 2012: {old_rmse_2012:.3f}")
    print(f"Retrained model RMSE on 2012: {new_rmse_2012:.3f}")

    retrained_path = os.path.join("models", "retrained_model.joblib")
    os.makedirs("models", exist_ok=True)
    joblib.dump(retrained_model, retrained_path)
    print("Saved retrained model:", retrained_path)

    # -------- MLflow logging (optional) --------
    if _HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("ML2_Task2_Drift_Monitoring")

            with mlflow.start_run(run_name="Task2_Data_Drift_Check"):
                mlflow.set_tag("role", "drift_monitoring")
                mlflow.set_tag("model_monitored", "best_model.joblib")

                mlflow.log_param("reference_dataset", "day_2011.csv")
                mlflow.log_param("current_dataset", "day_2012.csv")
                mlflow.log_param("num_features", X_ref.shape[1])

                mlflow.log_metric("rmse_2011", rmse_2011)
                mlflow.log_metric("mae_2011", mae_2011)
                mlflow.log_metric("rmse_2012", rmse_2012)
                mlflow.log_metric("mae_2012", mae_2012)
                mlflow.log_metric("rmse_change", rmse_2012 - rmse_2011)

                mlflow.log_metric("old_rmse_2012", old_rmse_2012)
                mlflow.log_metric("retrained_rmse_2012", new_rmse_2012)
                mlflow.log_metric("rmse_improvement_2012", old_rmse_2012 - new_rmse_2012)

                mlflow.log_artifact(summary_csv)
                mlflow.log_artifact(hum_plot_path)
                mlflow.log_artifact(drift_html_path)
                mlflow.log_artifact(drift_json_path)
                mlflow.log_artifact(retrained_path)
        except Exception as e:
            warnings.warn(f"MLflow logging skipped (MLflow server may be off). Details: {e}")

if __name__ == "__main__":
    main()
