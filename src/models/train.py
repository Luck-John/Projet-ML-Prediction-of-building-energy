import os
import sys
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

# Fixer tous les seeds pour reproductibilité maximale (déterminisme)
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)

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

from src.preprocessing.preprocessor import preprocess_data, save_processed_df
from src.features.engineer import engineer_features, prepare_for_training

# Config
DATA_URL = "https://raw.githubusercontent.com/MouslyDiaw/handson-machine-learning/refs/heads/master/data/2016_Building_Energy_Benchmarking.csv"
PROCESSED_DATA_PATH = "data/processed/seattle_energy_cleaned_final.csv"
MODEL_ARTIFACT = "artifacts/model.joblib"
ARTIFACTS_DIR = "artifacts"
RANDOM_STATE = 42
TARGET_COL = "SiteEnergyUse_log"


def train_model(use_energy_star: bool = True, mlflow_experiment: str = "energy_buildings"):
    """
    Complete training pipeline:
    1. Preprocess raw data (matching notebook exactly)
    2. Engineer features (geo, clustering, etc.)
    3. Save processed data with versioning
    4. Train stacking ensemble with GridSearch on base learners
    5. Save model + metadata
    """
    os.environ.setdefault('MLFLOW_TRACKING_URI', 'file:./mlruns')
    mlflow.set_experiment(mlflow_experiment)

    # ========== STEP 1: PREPROCESS DATA ==========
    print("=" * 60)
    print("STEP 1: PREPROCESSING")
    print("=" * 60)
    df = preprocess_data(DATA_URL)
    print(f"[OK] After preprocessing: {df.shape[0]} rows x {df.shape[1]} cols")
    
    # ========== STEP 2: ENGINEER FEATURES ==========
    print("\n" + "=" * 60)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 60)
    df = engineer_features(df)
    print(f"[OK] After feature engineering: {df.shape[0]} rows x {df.shape[1]} cols")
    
    # Remove duplicate column names (breaks LightGBM)
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if len(dup_cols) > 0:
        print(f"[WARN] Removing duplicate columns: {dup_cols}")
        df = df.loc[:, ~df.columns.duplicated()]
    
    # ========== STEP 3: SAVE PROCESSED DATA ==========
    print("\n" + "=" * 60)
    print("STEP 3: SAVING PROCESSED DATA")
    print("=" * 60)
    save_processed_df(df, 
                     output_path=PROCESSED_DATA_PATH,
                     version_path="artifacts/data_version.json",
                     force=True)
    
    # ========== STEP 4: PREPARE FOR MODELING ==========
    print("\n" + "=" * 60)
    print("STEP 4: TRAIN/TEST SPLIT")
    print("=" * 60)
    X, y = prepare_for_training(df)
    print(f"[OK] Features shape: {X.shape}")
    print(f"[OK] Target shape: {y.shape}")
    print(f"[OK] Training columns: {list(X.columns)}")
    
    # Save training columns for reference
    training_columns = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"[OK] Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")
    
    # ========== STEP 5: TARGET ENCODING ==========
    print("\n" + "=" * 60)
    print("STEP 5: TARGET ENCODING CATEGORICAL VARIABLES")
    print("=" * 60)
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(cat_cols) > 0:
        print(f"[OK] Encoding {len(cat_cols)} categorical columns")
        encoder = ce.TargetEncoder(cols=cat_cols, smoothing=10, handle_unknown='value')
        X_train_enc = encoder.fit_transform(X_train, y_train)
        X_test_enc = encoder.transform(X_test)
    else:
        print("[OK] No categorical columns to encode")
        encoder = None
        X_train_enc, X_test_enc = X_train.copy(), X_test.copy()
    
    # ========== STEP 6: GRID SEARCH BASE LEARNERS ==========
    print("\n" + "=" * 60)
    print("STEP 6: GRID SEARCH BASE LEARNERS")
    print("=" * 60)
    
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
    
    for name, (model_obj, params) in models_config.items():
        print(f"\n  GridSearchCV for {name}...")
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', model_obj)
        ])
        grid = GridSearchCV(
            pipe, params,
            cv=3,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            refit=True
        )
        grid.fit(X_train_enc, y_train)
        raw = grid.best_params_
        clean = {k.replace('model__', ''): v for k, v in raw.items()}
        best_params_storage[name] = clean
        print(f"  [OK] {name} best params: {clean}")
        print(f"      Best CV score: {grid.best_score_:.4f}")
    
    # ========== STEP 7: BUILD STACKING ENSEMBLE ==========
    print("\n" + "=" * 60)
    print("STEP 7: TRAINING STACKING ENSEMBLE")
    print("=" * 60)
    
    # Create base learners with best params
    base_learners = [
        ('et', ExtraTreesRegressor(
            random_state=RANDOM_STATE,
            **best_params_storage.get('ExtraTrees', {})
        )),
        ('xgb', XGBRegressor(
            random_state=RANDOM_STATE,
            objective='reg:squarederror',
            **best_params_storage.get('XGBoost', {})
        )),
        ('hgb', HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            **best_params_storage.get('HistGradientBoosting', {})
        )),
        ('lgbm', LGBMRegressor(
            random_state=RANDOM_STATE,
            verbose=-1,
            **best_params_storage.get('LightGBM', {})
        ))
    ]
    
    print(f"[OK] Building StackingRegressor with {len(base_learners)} base learners")
    print(f"     Meta-learner: LinearSVR(C=10, dual='auto')")
    
    final_stack = StackingRegressor(
        estimators=base_learners,
        final_estimator=LinearSVR(C=10, random_state=RANDOM_STATE, dual='auto', max_iter=10000),
        cv=5,
        n_jobs=-1,
        passthrough=False
    )
    
    print(f"[OK] Fitting StackingRegressor (this may take a few minutes)...")
    final_stack.fit(X_train_enc, y_train)
    print(f"[OK] StackingRegressor fitted successfully")
    
    # ========== STEP 8: EVALUATE MODEL ==========
    print("\n" + "=" * 60)
    print("STEP 8: MODEL EVALUATION")
    print("=" * 60)
    
    def evaluate_performance(model, X, y, y_raw, prefix=""):
        """Evaluate on raw scale (exp of log predictions)"""
        y_pred_log = model.predict(X)
        y_pred_real = np.exp(y_pred_log)
        y_real = np.exp(y_raw)
        
        mape = mean_absolute_percentage_error(y_real, y_pred_real)
        mae = mean_absolute_error(y_real, y_pred_real)
        r2 = r2_score(y_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_real, y_pred_real))
        
        metrics = {
            f"{prefix}MAPE": float(mape),
            f"{prefix}MAE": float(mae),
            f"{prefix}R2": float(r2),
            f"{prefix}RMSE": float(rmse)
        }
        return metrics
    
    metrics_train = evaluate_performance(final_stack, X_train_enc, y_train, y_train, "Train_")
    metrics_test = evaluate_performance(final_stack, X_test_enc, y_test, y_test, "Test_")
    
    print(f"\nTRAIN Metrics:")
    for k, v in metrics_train.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nTEST Metrics:")
    for k, v in metrics_test.items():
        print(f"  {k}: {v:.4f}")
    
    # ========== STEP 9: SAVE ARTIFACTS ==========
    print("\n" + "=" * 60)
    print("STEP 9: SAVING ARTIFACTS")
    print("=" * 60)
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    # Remove old files (prevent Windows locking)
    pkl_path = os.path.join(ARTIFACTS_DIR, 'model.pkl')
    best_params_path = os.path.join(ARTIFACTS_DIR, 'best_params.joblib')
    
    for old_file in [MODEL_ARTIFACT, pkl_path, best_params_path]:
        if os.path.exists(old_file):
            try:
                os.remove(old_file)
            except Exception as e:
                print(f"  [WARN] Could not remove {old_file}: {e}")
    
    # Load KMeans models (saved by engineer_features)
    kmeans_geo = joblib.load(os.path.join(ARTIFACTS_DIR, "kmeans_neighborhood.joblib"))
    kmeans_surf = joblib.load(os.path.join(ARTIFACTS_DIR, "kmeans_surface.joblib"))
    
    # Save main model
    model_dict = {
        'model': final_stack,
        'encoder': encoder,
        'kmeans_geo': kmeans_geo,
        'kmeans_surf': kmeans_surf,
        'training_columns': training_columns,
        'best_params': best_params_storage,
        'target_col': TARGET_COL
    }
    joblib.dump(model_dict, MODEL_ARTIFACT)
    print(f"[OK] Saved: {MODEL_ARTIFACT}")
    
    # Save best params
    joblib.dump(best_params_storage, best_params_path)
    print(f"[OK] Saved: {best_params_path}")
    
    # Also save as pickle
    import pickle
    with open(pkl_path, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"[OK] Saved: {pkl_path}")
    
    # ========== STEP 10: MLFLOW LOGGING ==========
    print("\n" + "=" * 60)
    print("STEP 10: MLFLOW LOGGING")
    print("=" * 60)
    
    try:
        with mlflow.start_run(run_name='stacking_final'):
            mlflow.log_params({
                'use_energy_star': use_energy_star,
                'random_state': RANDOM_STATE,
                'n_base_learners': len(base_learners),
                'meta_learner': 'LinearSVR'
            })
            for k, v in {**metrics_train, **metrics_test}.items():
                mlflow.log_metric(k, float(v))
            mlflow.log_artifact(MODEL_ARTIFACT)
            print(f"[OK] Logged to MLflow")
    except Exception as e:
        print(f"[WARN] MLflow logging skipped: {e}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {MODEL_ARTIFACT}")
    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")
    
    return final_stack, best_params_storage


if __name__ == '__main__':
    train_model(use_energy_star=True, mlflow_experiment="energy_buildings")
