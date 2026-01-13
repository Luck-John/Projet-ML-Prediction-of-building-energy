"""
API FastAPI pour la prédiction de consommation énergétique des bâtiments de Seattle
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")
API_VERSION = "1.0.0"
API_TITLE = "Seattle Energy Consumption Prediction API"
API_DESCRIPTION = "API de prédiction de consommation énergétique pour les bâtiments non-résidentiels de Seattle."

SEATTLE_CENTER_LAT = 47.6062
SEATTLE_CENTER_LON = -122.3321

# =====================================================
# SCHÉMAS PYDANTIC
# =====================================================

class BuildingInput(BaseModel):
    BuildingType: str
    PrimaryPropertyType: str
    ZipCode: str
    CouncilDistrictCode: str
    Neighborhood: str
    LargestPropertyUseType: str
    ListOfAllPropertyUseTypes: str
    NumberofBuildings: int
    NumberofFloors: float
    PropertyGFATotal: float
    PropertyGFAParking: float
    PropertyGFABuildings: float = Field(alias="PropertyGFABuilding(s)")
    LargestPropertyUseTypeGFA: float
    ENERGYSTARScore: float
    Latitude: float
    Longitude: float
    YearBuilt: int
    
    class Config:
        allow_population_by_field_name = True

class PredictionResponse(BaseModel):
    predicted_energy_consumption_kBtu: float
    predicted_energy_consumption_log: float
    building_age: int
    timestamp: str
    model_version: str

class BatchPredictionRequest(BaseModel):
    buildings: List[BuildingInput]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    api_version: str
    model_loaded: bool
    model_path: str
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str

# =====================================================
# GESTIONNAIRE DE MODÈLE
# =====================================================

class ModelManager:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.kmeans_geo = None      # <-- NOM MIS À JOUR
        self.kmeans_surf = None     # <-- NOM MIS À JOUR
        self.training_columns = None
        self.loaded = False
        
    def load_model(self, path: str = MODEL_PATH):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Fichier modèle non trouvé : {path}")
            
            model_dict = joblib.load(path)
            
            self.model = model_dict['model']
            self.encoder = model_dict['encoder']
            self.kmeans_geo = model_dict['kmeans_geo']          # <-- NOM MIS À JOUR
            self.kmeans_surf = model_dict['kmeans_surf']        # <-- NOM MIS À JOUR
            self.training_columns = model_dict['training_columns']
            self.loaded = True
            print(f"[OK] Modèle et artefacts chargés depuis {path}")
            
        except Exception as e:
            print(f"[ERREUR] Échec du chargement des artefacts : {e}")
            self.loaded = False
    
    def haversine_vectorized(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c
    
    def preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        
        # --- ÉTAPE 1: Feature Engineering ---
        df_processed['BuildingAge'] = 2016 - df_processed['YearBuilt']
        df_processed['PropertyGFATotal_log'] = np.log(df_processed['PropertyGFATotal'])
        df_processed['Distance_to_Center'] = self.haversine_vectorized(df_processed['Latitude'], df_processed['Longitude'], SEATTLE_CENTER_LAT, SEATTLE_CENTER_LON)
        df_processed['Is_Downtown'] = (df_processed['Distance_to_Center'] < 2).astype(int)
        angle_rad = np.radians(30)
        df_processed['Rotated_Lat'] = df_processed['Latitude'] * np.cos(angle_rad) - df_processed['Longitude'] * np.sin(angle_rad)
        df_processed['Rotated_Lon'] = df_processed['Latitude'] * np.sin(angle_rad) + df_processed['Longitude'] * np.cos(angle_rad)
        
        # --- ÉTAPE 2: Clustering ---
        df_processed['Neighborhood_Cluster'] = self.kmeans_geo.predict(df_processed[['Latitude', 'Longitude']]) # <-- NOM MIS À JOUR
        cluster_ids = self.kmeans_surf.predict(df_processed[['PropertyGFATotal_log']]) # <-- NOM MIS À JOUR
        df_processed['Surface_Cluster'] = "surf_group_" + cluster_ids.astype(str)
        
        # --- ÉTAPE 3: Alignement ---
        missing_cols = set(self.training_columns) - set(df_processed.columns)
        if missing_cols:
            raise ValueError(f"Colonnes manquantes après le feature engineering : {missing_cols}")
        df_processed = df_processed[self.training_columns]
        
        # --- ÉTAPE 4: Encodage ---
        for col in self.encoder.cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(str).str.strip().str.lower()
        df_encoded = self.encoder.transform(df_processed)
        
        # --- ÉTAPE 5: CORRECTION FINALE ---
        df_encoded['ZipCode'] = pd.to_numeric(df_encoded['ZipCode'], errors='coerce').fillna(0).astype(int)
        df_encoded['CouncilDistrictCode'] = pd.to_numeric(df_encoded['CouncilDistrictCode'], errors='coerce').fillna(0).astype(int)
            
        return df_encoded
    
    def predict(self, df: pd.DataFrame) -> tuple:
        if not self.loaded:
            raise RuntimeError("Le modèle n'est pas chargé.")
        
        df_processed = self.preprocess_input(df)
        y_pred_log = self.model.predict(df_processed)
        y_pred_real = np.exp(y_pred_log)
        
        return y_pred_real, y_pred_log

model_manager = ModelManager()

# =====================================================
# APPLICATION FASTAPI
# =====================================================

app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup_event():
    model_manager.load_model()

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(building: BuildingInput):
    if not model_manager.loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    
    try:
        df = pd.DataFrame([building.dict(by_alias=True)])
        y_pred_real, y_pred_log = model_manager.predict(df)
        
        return PredictionResponse(
            predicted_energy_consumption_kBtu=float(y_pred_real[0]),
            predicted_energy_consumption_log=float(y_pred_log[0]),
            building_age=2016 - building.YearBuilt,
            timestamp=datetime.utcnow().isoformat(),
            model_version=API_VERSION
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}")

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "API is running."}

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return HealthResponse(
        status="healthy" if model_manager.loaded else "unhealthy",
        api_version=API_VERSION,
        model_loaded=model_manager.loaded,
        model_path=MODEL_PATH,
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    if not model_manager.loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    
    try:
        df = pd.DataFrame([b.dict(by_alias=True) for b in request.buildings])
        y_pred_real, y_pred_log = model_manager.predict(df)
        
        predictions = [
            PredictionResponse(
                predicted_energy_consumption_kBtu=float(y_pred_real[i]),
                predicted_energy_consumption_log=float(y_pred_log[i]),
                building_age=2016 - b.YearBuilt,
                timestamp=datetime.utcnow().isoformat(),
                model_version=API_VERSION
            ) for i, b in enumerate(request.buildings)
        ]
        
        return BatchPredictionResponse(predictions=predictions, count=len(predictions), timestamp=datetime.utcnow().isoformat())
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Batch prediction failed: {str(e)}")

