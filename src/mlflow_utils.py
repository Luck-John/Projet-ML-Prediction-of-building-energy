"""MLflow configuration and tracking utilities."""

import os
import mlflow
from pathlib import Path

# MLflow tracking URI
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT_NAME = "building-energy-prediction"

def setup_mlflow():
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create or get experiment
    try:
        experiment_id = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME).experiment_id
    except:
        experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return experiment_id


def log_model_params(params: dict):
    """Log model hyperparameters to MLflow."""
    mlflow.log_params(params)


def log_model_metrics(metrics: dict):
    """Log model evaluation metrics to MLflow."""
    mlflow.log_metrics(metrics)


def log_model_artifact(model, artifact_name: str, format: str = "joblib"):
    """Log model artifact to MLflow."""
    import joblib
    import pickle
    
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    if format == "joblib":
        model_path = os.path.join(artifacts_dir, f"{artifact_name}.joblib")
        joblib.dump(model, model_path)
    elif format == "pickle":
        model_path = os.path.join(artifacts_dir, f"{artifact_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    mlflow.log_artifact(model_path, artifact_path="models")


def load_best_run(metric_name: str = "rmse", ascending: bool = True):
    """Load the best run based on a metric."""
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
    
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id])
    
    if not runs:
        return None
    
    # Sort by metric
    best_run = min(runs, key=lambda r: r.data.metrics.get(metric_name, float('inf')))
    return best_run


if __name__ == "__main__":
    setup_mlflow()
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
