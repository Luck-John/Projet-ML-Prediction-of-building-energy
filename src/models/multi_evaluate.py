import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_percentage_error, r2_score

import category_encoders as ce

from src.preprocessing.preprocessor import preprocess_data
from src.features.engineer import engineer_features

DATA_PATH = "data/processed/2016_Building_Energy_Benchmarking.csv"
ARTIFACT = "artifacts/multi_model_ranking.joblib"
RANDOM_STATE = 42
TARGET_COL = "SiteEnergyUse_log"


def make_models():
    models = {
        "ExtraTrees": ExtraTreesRegressor(random_state=RANDOM_STATE, n_estimators=200, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=200, n_jobs=-1),
        # linear models will be wrapped with a scaler in the pipeline later
        "LinearRegression": LinearRegression(),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "ElasticNet": ElasticNet(random_state=RANDOM_STATE),
        "KNN": KNeighborsRegressor(),
        "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "SVM_Linear": LinearSVR(random_state=RANDOM_STATE, max_iter=10000)
    }

    # optional non-sklearn models
    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMRegressor(random_state=RANDOM_STATE)
    except Exception:
        models['LightGBM'] = None

    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
    except Exception:
        models['XGBoost'] = None

    return models


def eval_model(name, model, X_train, X_test, y_train, y_test):
    result = dict(name=name)
    if model is None:
        result.update({'error': 'missing_dependency'})
        return result

    # find categorical cols
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(cat_cols) > 0:
        encoder = ce.TargetEncoder(cols=cat_cols, smoothing=10, handle_unknown='value')
        X_tr_enc = encoder.fit_transform(X_train.copy(), y_train)
        X_te_enc = encoder.transform(X_test.copy())
    else:
        encoder = None
        X_tr_enc = X_train.copy()
        X_te_enc = X_test.copy()

    imputer = SimpleImputer(strategy='median')
    X_tr_num = pd.DataFrame(imputer.fit_transform(X_tr_enc), columns=X_tr_enc.columns)
    X_te_num = pd.DataFrame(imputer.transform(X_te_enc), columns=X_te_enc.columns)

    # wrap scaler for linear / svm / knn models
    model_name = name.lower()
    if any(k in model_name for k in ['linearregression', 'lasso', 'ridge', 'elasticnet', 'svm', 'svm_linear', 'knn']):
        model = Pipeline([('scaler', StandardScaler()), ('model', model)])

    try:
        model.fit(X_tr_num, y_train)
    except Exception as e:
        result.update({'error': str(e)})
        return result

    y_pred_test = model.predict(X_te_num)
    y_pred_train = model.predict(X_tr_num)

    # real-unit metrics
    y_test_real = np.exp(y_test)
    y_pred_test_real = np.exp(y_pred_test)
    y_train_real = np.exp(y_train)
    y_pred_train_real = np.exp(y_pred_train)

    test_mape = mean_absolute_percentage_error(y_test_real, y_pred_test_real)
    test_r2_real = r2_score(y_test_real, y_pred_test_real)
    test_r2_log = r2_score(y_test, y_pred_test)
    train_r2_real = r2_score(y_train_real, y_pred_train_real)

    result.update({
        'Test_MAPE_Real': float(test_mape),
        'Test_R2_Real': float(test_r2_real),
        'Test_R2_Log': float(test_r2_log),
        'Train_R2_Real': float(train_r2_real),
        'fitted_model': model,
        'encoder': encoder,
        'imputer_cols': list(X_tr_num.columns)
    })
    return result


def run_scenario(use_energy_star: bool, save_artifact: bool = True):
    df = preprocess_data(DATA_PATH)
    df = engineer_features(df)

    if not use_energy_star and 'ENERGYSTARScore' in df.columns:
        df = df.drop(columns=['ENERGYSTARScore'])

    X = df.drop(columns=[c for c in ["SiteEnergyUse(kBtu)", TARGET_COL, "PropertyGFATotal"] if c in df.columns])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    models = make_models()
    results = []
    for name, model in models.items():
        print(f"Running {name} (use_energy_star={use_energy_star})...")
        res = eval_model(name, model, X_train, X_test, y_train, y_test)
        results.append(res)

    results_clean = [r for r in results if r.get('error') is None]
    results_sorted = sorted(results_clean, key=lambda r: r.get('Test_MAPE_Real', float('inf')))

    # save ranking artifact per scenario
    if save_artifact:
        os.makedirs(os.path.dirname(ARTIFACT), exist_ok=True)
        base = 'multi_model_ranking_with_score.joblib' if use_energy_star else 'multi_model_ranking_no_score.joblib'
        joblib.dump({'ranking': results_sorted, 'raw': results, 'use_energy_star': use_energy_star}, os.path.join('artifacts', base))

    return results_sorted


def run_both_and_save_best():
    best_models = {}
    for flag in [True, False]:
        ranking = run_scenario(use_energy_star=flag, save_artifact=True)
        if len(ranking) == 0:
            continue
        best = ranking[0]
        # save best fitted model for API
        model_artifact_name = 'best_model_with_score.joblib' if flag else 'best_model_no_score.joblib'
        to_save = {
            'model': best.get('fitted_model'),
            'encoder': best.get('encoder'),
            'imputer_cols': best.get('imputer_cols'),
            'metrics': {k: best[k] for k in ['Test_MAPE_Real', 'Test_R2_Real', 'Train_R2_Real']}
        }
        joblib.dump(to_save, os.path.join('artifacts', model_artifact_name))
        best_models[flag] = best

    return best_models


if __name__ == '__main__':
    bests = run_both_and_save_best()
    print('\nSaved best models for scenarios:')
    for flag, best in bests.items():
        fname = 'best_model_with_score.joblib' if flag else 'best_model_no_score.joblib'
        print(f"use_energy_star={flag}: {best['name']} saved -> artifacts/{fname}  MAPE={best['Test_MAPE_Real']:.4f}")
