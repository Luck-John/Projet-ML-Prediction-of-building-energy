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

