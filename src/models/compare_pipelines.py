import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import category_encoders as ce

from src.preprocessing.preprocessor import preprocess_data, preprocess_df
from src.features.engineer import engineer_features
from src.models.evaluate import evaluate_model

DATA_PATH = 'data/processed/2016_Building_Energy_Benchmarking.csv'
RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def train_and_eval(X_train, X_test, y_train, y_test, encode_with=None):
    # Optionally apply TargetEncoder (encode_with = list of cols)
    encoder = None
    if encode_with:
        encoder = ce.TargetEncoder(cols=encode_with, smoothing=10, handle_unknown='value')
        X_train = encoder.fit_transform(X_train, y_train)
        X_test = encoder.transform(X_test)

    # Imputer + model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', ExtraTreesRegressor(random_state=RANDOM_STATE, n_estimators=300))
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = evaluate_model(pipeline.named_steps['model'], X_test, y_test, y_test_is_log=True)
    return metrics, pipeline, encoder


def notebook_like_pipeline(df_raw):
    # Follow notebook's additional steps: outlier removal, logs already in engineer
    df = df_raw.copy()
    # Outlier filters as in notebook
    df = df[df['SiteEnergyUse(kBtu)'] < 2.0 * 10**8]
    df = df[df['PropertyGFATotal'] < 3.0 * 10**6]

    # feature engineering (adds log cols)
    df = engineer_features(df)

    # MultiLabelBinarizer for ListOfAllPropertyUseTypes -> create df_mlb
    list_col = 'ListOfAllPropertyUseTypes'
    if list_col in df.columns:
        temp_list = df[list_col].fillna('').astype(str).apply(lambda x: [i.strip().lower() for i in x.split(',')] if x != '' else [])
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        df_mlb = pd.DataFrame(mlb.fit_transform(temp_list), columns=[f"use_{c}" for c in mlb.classes_], index=df.index)
    else:
        df_mlb = pd.DataFrame(index=df.index)

    # standard categorical columns
    std_cat_cols = ['PrimaryPropertyType', 'BuildingType', 'Neighborhood', 'LargestPropertyUseType']
    std_cat_cols = [c for c in std_cat_cols if c in df.columns]

    # Build X containing numeric features + mlb columns + explicit categorical columns to encode
    X_num = df.select_dtypes(include=[np.number]).copy()
    X = pd.concat([X_num, df_mlb], axis=1)

    # add back the standard categorical columns (to be target-encoded later)
    for c in std_cat_cols:
        if c in df.columns:
            X[c] = df[c]

    # drop administrative columns if present
    exclude = ['SiteEnergyUse(kBtu)', 'SiteEnergyUse_log', 'PropertyGFATotal']
    X = X.drop(columns=[c for c in exclude if c in X.columns], errors='ignore')

    y = df['SiteEnergyUse_log']

    nb_cat_existing = [c for c in std_cat_cols if c in X.columns]
    return X, y, nb_cat_existing


def refactored_pipeline(df_raw):
    # Use preprocess_df (which doesn't remove the notebook outliers)
    df = preprocess_df(df_raw)
    df = engineer_features(df)

    # create X,y
    exclude = ['SiteEnergyUse(kBtu)', 'SiteEnergyUse_log', 'PropertyGFATotal']
    X = df.drop(columns=[c for c in exclude if c in df.columns], errors='ignore')
    y = df['SiteEnergyUse_log']
    # categorical columns for target encoding
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    return X, y, cat_cols


if __name__ == '__main__':
    df_raw = pd.read_csv(DATA_PATH)

    # Notebook-like
    X_nb, y_nb, nb_cat = notebook_like_pipeline(df_raw)
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_nb, y_nb, test_size=0.2, random_state=RANDOM_STATE)
    nb_metrics, nb_model, nb_encoder = train_and_eval(Xn_train, Xn_test, yn_train, yn_test, encode_with=nb_cat if nb_cat else None)

    # Refactored
    X_ref, y_ref, ref_cat = refactored_pipeline(df_raw)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_ref, y_ref, test_size=0.2, random_state=RANDOM_STATE)
    ref_metrics, ref_model, ref_encoder = train_and_eval(Xr_train, Xr_test, yr_train, yr_test, encode_with=ref_cat if ref_cat else None)

    print('\n--- Notebook-like pipeline metrics ---')
    for k, v in nb_metrics.items():
        print(f'{k}: {v:.6f}')

    print('\n--- Refactored pipeline metrics ---')
    for k, v in ref_metrics.items():
        print(f'{k}: {v:.6f}')

    # Save a small report (delete old one first to avoid lock issues)
    report_path = 'artifacts/compare_report.joblib'
    if os.path.exists(report_path):
        os.remove(report_path)
    report = {'notebook': nb_metrics, 'refactored': ref_metrics}
    joblib.dump(report, report_path)
    print(f'\nReport saved to {report_path}')
