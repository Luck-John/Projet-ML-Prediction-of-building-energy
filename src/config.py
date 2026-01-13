"""
Project configuration and constants.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# Create directories if they don't exist
for directory in [PROCESSED_DATA_DIR, ARTIFACTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data URLs
DATA_URL = "https://raw.githubusercontent.com/MouslyDiaw/handson-machine-learning/refs/heads/master/data/2016_Building_Energy_Benchmarking.csv"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature engineering
SEATTLE_CENTER_LAT = 47.6062
SEATTLE_CENTER_LON = -122.3321
KMEANS_CLUSTERS_NEIGHBORHOOD = 10
KMEANS_CLUSTERS_SURFACE = 2

# MLflow configuration
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT_NAME = "building-energy-prediction"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Model artifacts
PREPROCESSOR_ARTIFACTS = {
    "neighborhood_kmeans_joblib": "artifacts/kmeans_neighborhood.joblib",
    "neighborhood_kmeans_pkl": "artifacts/kmeans_neighborhood.pkl",
    "surface_kmeans_joblib": "artifacts/kmeans_surface.joblib",
    "surface_kmeans_pkl": "artifacts/kmeans_surface.pkl",
}

MODEL_ARTIFACTS = {
    "model_joblib": "artifacts/model.joblib",
    "model_pkl": "artifacts/model.pkl",
    "best_params_joblib": "artifacts/best_params.joblib",
    "data_version": "artifacts/data_version.json",
}

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Artifacts Directory: {ARTIFACTS_DIR}")
    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
