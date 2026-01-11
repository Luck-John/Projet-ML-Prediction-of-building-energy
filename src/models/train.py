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
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import HistGradientBoostingRegressor
import category_encoders as ce

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.preprocessing.preprocessor import preprocess_data, preprocess_df
from src.features.engineer import engineer_features

# Config
DATA_PATH = "data/processed/2016_Building_Energy_Benchmarking.csv"
MODEL_ARTIFACT = "artifacts/model.joblib"
ARTIFACTS_DIR = "artifacts"
RANDOM_STATE = 42
TARGET_COL = "SiteEnergyUse_log"


def prepare_xy(df: pd.DataFrame):
    """Préparation X et y pour la modélisation (notebook 11 - ligne 1190-1198)"""
    cols_to_exclude = [
        "SiteEnergyUse(kBtu)", "SiteEnergyUse_log",
        "EnergyIntensity", "EnergyIntensity_Log",
        "PropertyGFATotal",
        "OSEBuildingID", "DataYear",
        "ListOfAllPropertyUseTypes", "ListOfAllPropertyUseTypes_clean"
    ]
    X = df.drop(columns=[c for c in cols_to_exclude if c in df.columns], errors='ignore')
    y = df[TARGET_COL]
    return X, y


def evaluate_performance(model, X, y, prefix="Test"):
    y_pred = model.predict(X)
    y_pred_real = np.exp(y_pred)
    y_real = np.exp(y)
    mape = mean_absolute_percentage_error(y_real, y_pred_real)
    mae = mean_absolute_error(y_real, y_pred_real)
    r2 = r2_score(y_real, y_pred_real)
    # older sklearn versions may not accept `squared` kwarg
    rmse_raw = mean_squared_error(y_real, y_pred_real)
    rmse = float(np.sqrt(rmse_raw))
    metrics = {
        f"{prefix}_MAPE_Real": float(mape),
        f"{prefix}_MAE_Real": float(mae),
        f"{prefix}_R2_Real": float(r2),
        f"{prefix}_RMSE_Real": float(rmse)
    }
    return metrics


def train_model(use_energy_star: bool = True, mlflow_experiment: str = "energy_buildings"):
    """Pipeline principal d'entraînement reproduisant le notebook (11).
    Cette version entraîne les base learners, récupère leurs meilleurs
    hyperparamètres puis construit et entraîne le StackingRegressor final
    avec `LinearSVR(C=10)` comme meta-learner.
    """
    os.environ.setdefault('MLFLOW_TRACKING_URI', 'file:./mlruns')

    mlflow.set_experiment(mlflow_experiment)

    # 1. Load & preprocess
    print(">>> Loading data...")
    df = preprocess_data(DATA_PATH)
    print(f"Shape after preprocess: {df.shape}")

    # 2. Feature engineering
    print(">>> Feature engineering...")
    df = engineer_features(df)
    print(f"Shape after engineer: {df.shape}")

    # 3. Prepare X, y and split
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4. Target encoding
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(cat_cols) > 0:
        encoder = ce.TargetEncoder(cols=cat_cols, smoothing=10, handle_unknown='value')
        X_train_enc = encoder.fit_transform(X_train, y_train)
        X_test_enc = encoder.transform(X_test)
    else:
        encoder = None
        X_train_enc, X_test_enc = X_train.copy(), X_test.copy()

    # 5. Train base learners (grid search) - grids per notebook
    models_config = {
        'ExtraTrees': (
            ExtraTreesRegressor(random_state=RANDOM_STATE),
            {
                'model__n_estimators': [100, 300, 500],
                'model__max_depth': [None, 10, 20]
            }
        ),
        'XGBoost': (
            XGBRegressor(random_state=RANDOM_STATE, objective='reg:squarederror'),
            {
                'model__n_estimators': [100, 300],
                'model__learning_rate': [0.01, 0.05],
                'model__max_depth': [3, 6]
            }
        ),
        'LightGBM': (
            LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
            {
                'model__n_estimators': [100, 300],
                'model__learning_rate': [0.01, 0.05],
                'model__num_leaves': [31, 50]
            }
        ),
        'HistGradientBoosting': (
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                'model__learning_rate': [0.01, 0.05],
                'model__max_iter': [100, 200]
            }
        )
    }

    best_params_storage = {}
    trained_estimators = {}

    for name, (model_obj, params) in models_config.items():
        print(f">>> GridSearch for {name}...")
        pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', model_obj)])
        grid = GridSearchCV(pipe, params, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1, refit=True)
        grid.fit(X_train_enc, y_train)
        best = grid.best_estimator_
        # store cleaned params
        raw = grid.best_params_
        clean = {k.replace('model__', ''): v for k, v in raw.items()}
        best_params_storage[name] = clean
        trained_estimators[name] = best.named_steps['model']
        print(f"-> {name} best params: {clean}")

    # 6. Build base_learners for stacking using best params
    base_learners = []
    # ExtraTrees
    et_params = best_params_storage.get('ExtraTrees', {})
    base_learners.append(('et', ExtraTreesRegressor(random_state=RANDOM_STATE, **et_params)))
    # XGBoost
    xgb_params = best_params_storage.get('XGBoost', {})
    base_learners.append(('xgb', XGBRegressor(random_state=RANDOM_STATE, objective='reg:squarederror', **xgb_params)))
    # HistGradient
    hgb_params = best_params_storage.get('HistGradientBoosting', {})
    base_learners.append(('hgb', HistGradientBoostingRegressor(random_state=RANDOM_STATE, **hgb_params)))
    # LightGBM
    lgbm_params = best_params_storage.get('LightGBM', {})
    base_learners.append(('lgbm', LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, **lgbm_params)))

    print(">>> Training final stacking model (this can take time)...")
    final_stack = StackingRegressor(
        estimators=base_learners,
        final_estimator=LinearSVR(C=10, random_state=RANDOM_STATE, dual='auto', max_iter=10000),
        cv=5,
        n_jobs=-1,
        passthrough=False
    )

    final_stack.fit(X_train_enc, y_train)

    # Evaluate final
    metrics_train = evaluate_performance(final_stack, X_train_enc, y_train, "Train")
    metrics_test = evaluate_performance(final_stack, X_test_enc, y_test, "Test")

    print(f"Final Test MAPE: {metrics_test['Test_MAPE_Real']:.4f}")

    # 7. Save artifacts
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model_dict = {
        'model': final_stack,
        'encoder': encoder,
        'best_params': best_params_storage,
        'target_col': TARGET_COL
    }
    joblib.dump(model_dict, MODEL_ARTIFACT)
    joblib.dump(best_params_storage, os.path.join(ARTIFACTS_DIR, 'best_params.joblib'))

    # Try MLflow logging (non-fatal)
    try:
        with mlflow.start_run(run_name='stacking_final'):
            mlflow.log_params({'use_energy_star': use_energy_star, 'random_state': RANDOM_STATE})
            for k, v in metrics_test.items():
                mlflow.log_metric(k, float(v))
            mlflow.log_artifact(MODEL_ARTIFACT)
    except Exception as e:
        print(f"⚠️ MLflow logging skipped: {e}")

    print(f"\n✅ Final stacking model saved to {MODEL_ARTIFACT}")
    return final_stack, best_params_storage


if __name__ == '__main__':
    train_model(use_energy_star=True, mlflow_experiment="energy_buildings")
