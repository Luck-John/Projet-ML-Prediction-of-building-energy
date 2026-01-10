import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
import category_encoders as ce

from src.preprocessing.preprocessor import preprocess_data, preprocess_df
from src.features.engineer import engineer_features

# Config
DATA_PATH = "data/processed/2016_Building_Energy_Benchmarking.csv"
MODEL_ARTIFACT = "artifacts/model.joblib"
RANDOM_STATE = 42
TARGET_COL = "SiteEnergyUse_log"


def prepare_xy(df: pd.DataFrame):
    cols_to_exclude = [
        "SiteEnergyUse(kBtu)",
        "SiteEnergyUse_log",
        "PropertyGFATotal"
    ]
    X = df.drop(columns=[c for c in cols_to_exclude if c in df.columns])
    y = df[TARGET_COL]
    return X, y


def train_model(use_energy_star: bool = True, mlflow_experiment: str = "default"):
    """Charge les données, applique preprocessing + feature engineering,
    entraîne un modèle et logge les métriques + artefacts dans MLflow.
    """

    mlflow.set_experiment(mlflow_experiment)

    df = preprocess_data(DATA_PATH)
    df = engineer_features(df)

    if not use_energy_star and "ENERGYSTARScore" in df.columns:
        df = df.drop(columns=["ENERGYSTARScore"])

    X, y = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Target encoding des colonnes catégorielles (fit sur train seulement)
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(cat_cols) > 0:
        encoder = ce.TargetEncoder(cols=cat_cols, smoothing=10, handle_unknown='value')
        X_train = encoder.fit_transform(X_train, y_train)
        X_test = encoder.transform(X_test)
    else:
        encoder = None

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(random_state=RANDOM_STATE, n_estimators=300, n_jobs=-1))
    ])

    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20]
    }

    # Use neg_mean_squared_error for GridSearchCV and refit on best score
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, refit=True)

    with mlflow.start_run(run_name=f"train_use_energy_star_{use_energy_star}"):
        # Log encoder information
        mlflow.log_param("cat_cols_count", int(len(cat_cols)))
        mlflow.log_param("target_encoder_smoothing", 10 if encoder is not None else None)

        grid.fit(X_train, y_train)

        best = grid.best_estimator_
        y_pred = best.predict(X_test)

        # back-transform if target is log
        y_pred_real = np.exp(y_pred)
        y_test_real = np.exp(y_test)

        mape = mean_absolute_percentage_error(y_test_real, y_pred_real)
        mae = mean_absolute_error(y_test_real, y_pred_real)
        r2 = r2_score(y_test_real, y_pred_real)

        mlflow.log_param("use_energy_star", use_energy_star)
        mlflow.log_param("best_params", grid.best_params_)

        mlflow.log_metric("mape", float(mape))
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("r2", float(r2))

        # save model artifact
        os.makedirs(os.path.dirname(MODEL_ARTIFACT), exist_ok=True)
        joblib.dump({
            "model": best,
            "encoder": encoder,
            "use_energy_star": use_energy_star
        }, MODEL_ARTIFACT)
        mlflow.log_artifact(MODEL_ARTIFACT)

    return best


if __name__ == '__main__':
    train_model(use_energy_star=True, mlflow_experiment="energy_buildings")
