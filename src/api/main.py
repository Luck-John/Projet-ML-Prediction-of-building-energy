import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# API FASTAPI - TEMPLATE PRÊT À UTILISER
# ============================================================================
# Fichier: src/api/main.py
# À créer et utiliser pour une API REST

"""
DÉMARRER L'API:
    uvicorn src.api.main:app --reload

TESTER:
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{"PropertyGFATotal": 50000, "YearBuilt": 2000, ...}'
"""

from fastapi import FastAPI
from pydantic import BaseModel
import logging

app = FastAPI(
    title="Building Energy Prediction API",
    description="Prédire la consommation énergétique des bâtiments Seattle",
    version="1.0.0"
)

# ============================================================================
# Charger le modèle au démarrage
# ============================================================================

MODEL_PATH = Path("artifacts/model.joblib")

@app.on_event("startup")
async def load_model():
    """Charger le modèle au démarrage du serveur"""
    global model, encoder, best_params
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model_dict = joblib.load(MODEL_PATH)
    model = model_dict['model']
    encoder = model_dict['encoder']
    best_params = model_dict['best_params']
    
    logging.info("✅ Model loaded successfully")

# ============================================================================
# Routes
# ============================================================================

class PredictionRequest(BaseModel):
    """Schéma d'entrée pour les prédictions"""
    PropertyGFATotal: float
    YearBuilt: int
    PrimaryPropertyType: str
    ENERGYSTARScore: float
    # Ajoute d'autres features selon ton modèle
    
    class Config:
        example = {
            "PropertyGFATotal": 50000,
            "YearBuilt": 2005,
            "PrimaryPropertyType": "Office",
            "ENERGYSTARScore": 75.0,
        }

@app.get("/health")
def health_check():
    """Vérifier que l'API est en ligne"""
    return {
        "status": "✅ OK",
        "model": "StackingRegressor",
        "base_learners": ["ExtraTrees", "XGBoost", "LightGBM", "HistGradientBoosting"],
        "meta_learner": "LinearSVR(C=10)"
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    """Prédire la consommation énergétique"""
    try:
        # Créer DataFrame
        X = pd.DataFrame([request.dict()])
        
        # Encoder si nécessaire
        if encoder:
            X = encoder.transform(X)
        
        # Prédiction en log scale
        pred_log = model.predict(X)[0]
        
        # Convertir en scale réelle
        pred_real = np.exp(pred_log)
        
        return {
            "prediction_kBtu": float(pred_real),
            "prediction_log": float(pred_log),
            "confidence": "high" if 0 < pred_real < 1e8 else "low"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/metrics")
def get_model_metrics():
    """Retourner les métriques du modèle"""
    return {
        "model_type": "StackingRegressor",
        "test_mape": 0.4201,
        "test_r2": 0.527,
        "test_rmse": 7877872,
        "best_params": best_params,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
