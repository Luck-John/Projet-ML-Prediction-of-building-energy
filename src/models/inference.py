import os
import joblib
import numpy as np
import pandas as pd
from typing import Union

from src.preprocessing.preprocessor import preprocess_data, preprocess_df
from src.features.engineer import engineer_features

MODEL_ARTIFACT = "artifacts/model.joblib"


def load_model(path: str = MODEL_ARTIFACT):
    """Charge un modèle joblib sauvegardé."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare X comme lors de l'entraînement (suppression cible / colonnes volatiles)."""
    cols_to_exclude = [
        "SiteEnergyUse(kBtu)",
        "SiteEnergyUse_log",
        "PropertyGFATotal"
    ]
    return df.drop(columns=[c for c in cols_to_exclude if c in df.columns])


def predict(model, X: Union[pd.DataFrame, str], apply_engineering: bool = False) -> np.ndarray:
    """Fait des prédictions.

    - `X` peut être un DataFrame déjà prétraité ou un chemin vers un fichier CSV brut.
    - Si `apply_engineering` est True, on applique `engineer_features` après le preprocessing.
    Retourne la prédiction en unité réelle (exponentielle si le modèle prédit le log).
    """
    if isinstance(X, str):
        df = preprocess_data(X)
    elif isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        raise ValueError("X must be a pandas DataFrame or a path to a CSV file")

    if apply_engineering:
        df = engineer_features(df)

    X_prepared = _prepare_X(df)

    y_pred_log = model.predict(X_prepared)
    # si le modèle prédit la cible en log, on retransforme
    try:
        y_pred_real = np.exp(y_pred_log)
    except Exception:
        # si prediction non-log, renvoyer tel quel
        y_pred_real = y_pred_log

    return y_pred_real


if __name__ == '__main__':
    # Exécutable simple pour tester le chargement
    model = load_model()
    print("Modèle chargé :", type(model))
