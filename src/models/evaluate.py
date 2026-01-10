import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from typing import Dict


def _safe_exp(arr):
    try:
        return np.exp(arr)
    except Exception:
        return arr


def metrics_real(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcule MAE, RMSE, R2, MAPE sur valeurs réelles."""
    mae = mean_absolute_error(y_true, y_pred)
    # sklearn versions differ on `squared` argument; compute RMSE via sqrt of MSE
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "mape": float(mape)}


def evaluate_model(model, X_test, y_test, y_test_is_log: bool = True) -> Dict[str, float]:
    """Évalue un modèle sklearn-like.

    Si `y_test_is_log` est True, on convertit `y_test` et `y_pred` en valeurs réelles
    avant le calcul des métriques.
    """
    y_pred = model.predict(X_test)

    if y_test_is_log:
        y_test_real = _safe_exp(y_test)
        y_pred_real = _safe_exp(y_pred)
    else:
        y_test_real = y_test
        y_pred_real = y_pred

    return metrics_real(y_test_real, y_pred_real)


if __name__ == '__main__':
    print('Module d\'évaluation prêt.')
