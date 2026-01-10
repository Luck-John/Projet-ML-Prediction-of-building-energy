import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.preprocessing.preprocessor import preprocess_data
from src.features.engineer import engineer_features
from src.models.evaluate import evaluate_model

DATA_PATH = "data/processed/2016_Building_Energy_Benchmarking.csv"
MODEL_ARTIFACT = "artifacts/model.joblib"
RANDOM_STATE = 42


def prepare_xy(df):
    cols_to_exclude = [
        "SiteEnergyUse(kBtu)",
        "SiteEnergyUse_log",
        "PropertyGFATotal"
    ]
    X = df.drop(columns=[c for c in cols_to_exclude if c in df.columns])
    y = df['SiteEnergyUse_log']
    return X, y


if __name__ == '__main__':
    df = preprocess_data(DATA_PATH)
    df = engineer_features(df)

    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    if not os.path.exists(MODEL_ARTIFACT):
        raise FileNotFoundError(MODEL_ARTIFACT)

    artefact = joblib.load(MODEL_ARTIFACT)
    model = artefact.get('model', artefact)
    encoder = artefact.get('encoder', None)

    if encoder is not None:
        X_test = encoder.transform(X_test)

    metrics = evaluate_model(model, X_test, y_test, y_test_is_log=True)
    print('Evaluation metrics (on test set):')
    for k, v in metrics.items():
        print(f"- {k}: {v:.4f}")
