"""
API FastAPI pour la prédiction de consommation énergétique des bâtiments de Seattle
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
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
API_DESCRIPTION = """
API de prédiction de consommation énergétique pour les bâtiments non-résidentiels de Seattle.

## Fonctionnalités
* **Prédiction unique** : Prédire la consommation d'un bâtiment
* **Prédiction batch** : Prédire pour plusieurs bâtiments simultanément
* **Health check** : Vérifier le statut de l'API et du modèle

## Modèle
Stacking Regressor avec ExtraTrees, XGBoost, LightGBM et HistGradientBoosting.
"""

# Constantes géographiques
SEATTLE_CENTER_LAT = 47.6062
SEATTLE_CENTER_LON = -122.3321

# =====================================================
# SCHÉMAS PYDANTIC
# =====================================================

class BuildingInput(BaseModel):
    """Schéma pour les données d'entrée d'un bâtiment"""
    
    # Categorical features
    BuildingType: str = Field(..., description="Type de bâtiment")
    PrimaryPropertyType: str = Field(..., description="Type de propriété principal")
    ZipCode: str = Field(..., description="Code postal")
    CouncilDistrictCode: str = Field(..., description="Code du district")
    Neighborhood: str = Field(..., description="Quartier")
    LargestPropertyUseType: str = Field(..., description="Type d'usage principal")
    ListOfAllPropertyUseTypes: str = Field(..., description="Liste des types d'usage (séparés par virgules)")
    
    # Numeric features
    NumberofBuildings: int = Field(..., ge=1, description="Nombre de bâtiments")
    NumberofFloors: float = Field(..., ge=0, description="Nombre d'étages")
    PropertyGFATotal: float = Field(..., gt=0, description="Surface totale en pieds carrés")
    PropertyGFAParking: float = Field(..., ge=0, description="Surface de parking")
    PropertyGFABuildings: float = Field(..., ge=0, description="Surface des bâtiments", alias="PropertyGFABuilding(s)")
    LargestPropertyUseTypeGFA: float = Field(..., ge=0, description="Surface du type d'usage principal")
    ENERGYSTARScore: float = Field(..., ge=0, le=100, description="Score ENERGY STAR (0-100)")
    
    # Geographic features
    Latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    Longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    
    # Temporal feature
    YearBuilt: int = Field(..., ge=1800, le=2016, description="Année de construction")
    
    class Config:
        schema_extra = {
            "example": {
                "BuildingType": "NonResidential",
                "PrimaryPropertyType": "Office",
                "ZipCode": "98101",
                "CouncilDistrictCode": "7",
                "Neighborhood": "DOWNTOWN",
                "LargestPropertyUseType": "Office",
                "ListOfAllPropertyUseTypes": "Office,Parking",
                "NumberofBuildings": 1,
                "NumberofFloors": 15.0,
                "PropertyGFATotal": 250000.0,
                "PropertyGFAParking": 50000.0,
                "PropertyGFABuilding(s)": 200000.0,
                "LargestPropertyUseTypeGFA": 200000.0,
                "ENERGYSTARScore": 75.0,
                "Latitude": 47.6062,
                "Longitude": -122.3321,
                "YearBuilt": 2000
            }
        }
        allow_population_by_field_name = True

    @validator('PropertyGFATotal')
    def validate_total_surface(cls, v, values):
        """Validation: Surface totale doit être cohérente"""
        if v > 3_000_000:
            raise ValueError("PropertyGFATotal trop élevé (outlier détecté)")
        return v


class PredictionResponse(BaseModel):
    """Schéma pour la réponse de prédiction"""
    predicted_energy_consumption_kBtu: float = Field(..., description="Consommation prédite en kBtu")
    predicted_energy_consumption_log: float = Field(..., description="Consommation prédite (échelle log)")
    building_age: int = Field(..., description="Âge du bâtiment (calculé)")
    timestamp: str = Field(..., description="Timestamp de la prédiction")
    model_version: str = Field(..., description="Version du modèle")


class BatchPredictionRequest(BaseModel):
    """Schéma pour les prédictions batch"""
    buildings: List[BuildingInput] = Field(..., min_items=1, max_items=100)


class BatchPredictionResponse(BaseModel):
    """Schéma pour la réponse batch"""
    predictions: List[PredictionResponse]
    count: int
    timestamp: str


class HealthResponse(BaseModel):
    """Schéma pour le health check"""
    status: str
    api_version: str
    model_loaded: bool
    model_path: str
    timestamp: str


class ErrorResponse(BaseModel):
    """Schéma pour les erreurs"""
    error: str
    detail: Optional[str] = None
    timestamp: str


# =====================================================
# CHARGEMENT DU MODÈLE
# =====================================================

class ModelManager:
    """Gestionnaire du modèle ML"""
    
    def __init__(self):
        self.model = None
        self.encoder = None
        self.best_params = None
        self.target_col = None
        self.loaded = False
        
    def load_model(self, path: str = MODEL_PATH):
        """Charge le modèle depuis le fichier joblib"""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            
            model_dict = joblib.load(path)
            self.model = model_dict['model']
            self.encoder = model_dict.get('encoder')
            self.best_params = model_dict.get('best_params', {})
            self.target_col = model_dict.get('target_col', 'SiteEnergyUse_log')
            self.loaded = True
            
            print(f"[OK] Model loaded from {path}")
            if self.encoder:
                print(f"[OK] Encoder loaded (cols: {self.encoder.cols if hasattr(self.encoder, 'cols') else 'unknown'})")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.loaded = False
            return False
    
    def haversine_vectorized(self, lat1, lon1, lat2, lon2):
        """Calcule la distance Haversine entre deux points"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c
    
    def preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prétraite les données d'entrée pour correspondre EXACTEMENT au pipeline d'entraînement.
        
        ORDRE DES OPÉRATIONS (CRITIQUE):
        1. Normaliser les catégories (lowercase)
        2. Créer toutes les features numériques
        3. Encoder UNIQUEMENT les colonnes catégorielles avec TargetEncoder
        4. Retourner le DataFrame final avec toutes les features
        """
        df = df.copy()
        
        # ÉTAPE 1: Normaliser les colonnes catégorielles (AVANT l'encodage)
        categorical_cols = ['BuildingType', 'PrimaryPropertyType', 'ZipCode', 
                           'CouncilDistrictCode', 'Neighborhood', 
                           'LargestPropertyUseType', 'ListOfAllPropertyUseTypes']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        
        # ÉTAPE 2: Créer toutes les features numériques/engineered
        
        # 2.1 BuildingAge
        df['BuildingAge'] = 2016 - df['YearBuilt']
        
        # 2.2 Log transform de la surface
        df['PropertyGFATotal_log'] = np.log(df['PropertyGFATotal'])
        
        # 2.3 Distance au centre de Seattle
        df['Distance_to_Center'] = self.haversine_vectorized(
            df['Latitude'], df['Longitude'],
            SEATTLE_CENTER_LAT, SEATTLE_CENTER_LON
        )
        
        # 2.4 Is_Downtown (< 2 km du centre)
        df['Is_Downtown'] = (df['Distance_to_Center'] < 2).astype(int)
        
        # 2.5 Coordonnées tournées (30 degrés)
        angle_rad = np.radians(30)
        df['Rotated_Lat'] = df['Latitude'] * np.cos(angle_rad) - df['Longitude'] * np.sin(angle_rad)
        df['Rotated_Lon'] = df['Latitude'] * np.sin(angle_rad) + df['Longitude'] * np.cos(angle_rad)
        
        # 2.6 Neighborhood_Cluster
        from sklearn.cluster import KMeans
        if len(df) >= 10:
            kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
            df['Neighborhood_Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])
        else:
            df['Neighborhood_Cluster'] = 0
        
        # 2.7 Surface_Cluster
        if len(df) >= 2:
            kmeans_surf = KMeans(n_clusters=2, random_state=42, n_init=10)
            cluster_ids = kmeans_surf.fit_predict(df[['PropertyGFATotal_log']])
            df['Surface_Cluster'] = "surf_group_" + cluster_ids.astype(str)
        else:
            df['Surface_Cluster'] = "surf_group_0"
        
        # ÉTAPE 3: Sélectionner les colonnes finales
        final_columns = [
            'BuildingType', 'PrimaryPropertyType', 'ZipCode', 'CouncilDistrictCode',
            'Neighborhood', 'NumberofBuildings', 'NumberofFloors', 'PropertyGFATotal',
            'PropertyGFAParking', 'PropertyGFABuilding(s)', 'ListOfAllPropertyUseTypes',
            'LargestPropertyUseType', 'LargestPropertyUseTypeGFA', 'ENERGYSTARScore',
            'BuildingAge', 'PropertyGFATotal_log', 'Distance_to_Center', 
            'Neighborhood_Cluster', 'Is_Downtown', 'Rotated_Lat', 'Rotated_Lon',
            'Surface_Cluster'
        ]
        
        missing_cols = [col for col in final_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns after preprocessing: {missing_cols}")
        
        df_final = df[final_columns].copy()
        
        print(f"[DEBUG] Shape before encoding: {df_final.shape}")
        
        # ÉTAPE 4: Appliquer le TargetEncoder sur TOUTES les colonnes catégorielles
        # L'encoder a été fit sur toutes les colonnes object dans train.py
        if self.encoder is not None:
            # Identifier toutes les colonnes catégorielles
            cat_cols_in_final = df_final.select_dtypes(include=['object']).columns.tolist()
            
            print(f"[DEBUG] Categorical columns to encode: {cat_cols_in_final}")
            
            if cat_cols_in_final:
                # L'encoder attend ces colonnes dans l'ordre
                df_final[cat_cols_in_final] = self.encoder.transform(df_final[cat_cols_in_final])
                print(f"[DEBUG] Encoding complete")
        
        # ÉTAPE 5: Gérer les colonnes catégorielles non-encodées
        # Ces colonnes doivent rester des strings pour que l'encoder les gère
        # L'encoder a été fit sur TOUTES les colonnes object, pas juste 4
        # Donc on laisse toutes les colonnes catégorielles en string
        
        print(f"[DEBUG] Categorical columns (as strings): {df_final.select_dtypes(include=['object']).columns.tolist()}")
        
        
        print(f"[DEBUG] Final shape before model: {df_final.shape}")
        print(f"[DEBUG] Final dtypes:\n{df_final.dtypes}")
        
        return df_final
    
    def predict(self, df: pd.DataFrame) -> tuple:
        """Fait une prédiction"""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        # Preprocess (inclut l'encodage)
        df_processed = self.preprocess_input(df)
        
        # Predict (log scale)
        y_pred_log = self.model.predict(df_processed)
        
        # Convert back to original scale
        y_pred_real = np.exp(y_pred_log)
        
        return y_pred_real, y_pred_log


# Initialiser le gestionnaire de modèle
model_manager = ModelManager()


# =====================================================
# APPLICATION FASTAPI
# =====================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# ÉVÉNEMENTS DE DÉMARRAGE/ARRÊT
# =====================================================

@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage de l'API"""
    print("=" * 60)
    print(f"Starting {API_TITLE} v{API_VERSION}")
    print("=" * 60)
    success = model_manager.load_model()
    if not success:
        print("[WARNING] API started without model loaded!")
    else:
        print("[OK] API ready to serve predictions")


@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage lors de l'arrêt"""
    print("Shutting down API...")


# =====================================================
# ROUTES
# =====================================================

@app.get("/", tags=["General"])
async def root():
    """Route racine"""
    return {
        "message": "Seattle Energy Consumption Prediction API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Vérifie le statut de l'API et du modèle"""
    return HealthResponse(
        status="healthy" if model_manager.loaded else "unhealthy",
        api_version=API_VERSION,
        model_loaded=model_manager.loaded,
        model_path=MODEL_PATH,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(building: BuildingInput):
    """
    Prédit la consommation énergétique pour un seul bâtiment
    """
    try:
        if not model_manager.loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        # Convertir en DataFrame
        input_dict = building.dict(by_alias=True)
        df = pd.DataFrame([input_dict])
        
        print(f"[DEBUG] Input shape: {df.shape}")
        print(f"[DEBUG] Input columns: {list(df.columns)}")
        
        # Prédire
        y_pred_real, y_pred_log = model_manager.predict(df)
        
        # Calculer l'âge du bâtiment
        building_age = 2016 - building.YearBuilt
        
        return PredictionResponse(
            predicted_energy_consumption_kBtu=float(y_pred_real[0]),
            predicted_energy_consumption_log=float(y_pred_log[0]),
            building_age=building_age,
            timestamp=datetime.utcnow().isoformat(),
            model_version=API_VERSION
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Prédit la consommation énergétique pour plusieurs bâtiments
    """
    try:
        if not model_manager.loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        # Convertir en DataFrame
        input_dicts = [b.dict(by_alias=True) for b in request.buildings]
        df = pd.DataFrame(input_dicts)
        
        # Prédire
        y_pred_real, y_pred_log = model_manager.predict(df)
        
        # Construire les réponses
        predictions = []
        for i, building in enumerate(request.buildings):
            building_age = 2016 - building.YearBuilt
            predictions.append(
                PredictionResponse(
                    predicted_energy_consumption_kBtu=float(y_pred_real[i]),
                    predicted_energy_consumption_log=float(y_pred_log[i]),
                    building_age=building_age,
                    timestamp=datetime.utcnow().isoformat(),
                    model_version=API_VERSION
                )
            )
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Retourne les informations sur le modèle chargé"""
    if not model_manager.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_type": "StackingRegressor",
        "base_learners": ["ExtraTrees", "XGBoost", "LightGBM", "HistGradientBoosting"],
        "meta_learner": "LinearSVR",
        "target_variable": model_manager.target_col,
        "best_params": model_manager.best_params,
        "encoder": "TargetEncoder" if model_manager.encoder else None,
        "encoder_cols": model_manager.encoder.cols if (model_manager.encoder and hasattr(model_manager.encoder, 'cols')) else None
    }


# =====================================================
# EXCEPTION HANDLERS
# =====================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handler pour les HTTPException"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handler pour les exceptions générales"""
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )


# =====================================================
# POINT D'ENTRÉE
# =====================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )