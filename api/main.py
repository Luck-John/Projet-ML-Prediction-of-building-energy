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
from fastapi.responses import JSONResponse, FileResponse
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
    
    @validator('PropertyGFATotal', 'NumberofFloors', 'PropertyGFAParking', 
               'PropertyGFABuildings', 'LargestPropertyUseTypeGFA', 
               'ENERGYSTARScore', 'Latitude', 'Longitude', pre=True, always=True)
    def ensure_numeric(cls, v):
        """Convertir les valeurs en float, même si None ou vide"""
        if v is None or v == '':
            return 0.0
        if isinstance(v, str):
            try:
                return float(v.strip())
            except (ValueError, AttributeError):
                return 0.0
        return float(v)
    
    @validator('NumberofBuildings', 'YearBuilt', pre=True, always=True)
    def ensure_integer(cls, v):
        """Convertir les valeurs en int, même si None ou vide"""
        if v is None or v == '':
            return 0
        if isinstance(v, str):
            try:
                return int(float(v.strip()))
            except (ValueError, AttributeError):
                return 0
        return int(v)
    
    @validator('BuildingType', 'PrimaryPropertyType', 'ZipCode', 'CouncilDistrictCode',
               'Neighborhood', 'LargestPropertyUseType', 'ListOfAllPropertyUseTypes', 
               pre=True, always=True)
    def ensure_string(cls, v):
        """Convertir en string et nettoyer"""
        if v is None:
            return ""
        return str(v).strip()
    
    class Config:
        # CRITIQUE : permettre l'utilisation des alias ET des noms de champs
        populate_by_name = True
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
        self.kmeans_geo = None
        self.kmeans_surf = None
        self.training_columns = None
        self.loaded = False
        
    def load_model(self, path: str = MODEL_PATH):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Fichier modèle non trouvé : {path}")
            
            model_dict = joblib.load(path)
            
            self.model = model_dict['model']
            self.encoder = model_dict['encoder']
            self.kmeans_geo = model_dict['kmeans_geo']
            self.kmeans_surf = model_dict['kmeans_surf']
            self.training_columns = model_dict['training_columns']
            self.loaded = True
            print(f"[OK] Modèle et artefacts chargés depuis {path}")
            
        except Exception as e:
            print(f"[ERREUR] Échec du chargement des artefacts : {e}")
            self.loaded = False
    
    def haversine_vectorized(self, lat1, lon1, lat2, lon2):
        """Calcul de distance haversine vectorisé"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c
    
    def preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prétraitement des données avec conversion de types explicite"""
        df_processed = df.copy()
        
        # === CONVERSION DES TYPES NUMÉRIQUES ===
        numeric_columns = [
            'NumberofBuildings', 'NumberofFloors', 'PropertyGFATotal', 
            'PropertyGFAParking', 'PropertyGFABuilding(s)', 
            'LargestPropertyUseTypeGFA', 'ENERGYSTARScore', 
            'Latitude', 'Longitude', 'YearBuilt'
        ]
        
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
        
        # Vérifier les valeurs manquantes critiques
        critical_cols = ['PropertyGFATotal', 'Latitude', 'Longitude', 'YearBuilt']
        if df_processed[critical_cols].isnull().any().any():
            missing_info = df_processed[critical_cols].isnull().sum()
            raise ValueError(f"Valeurs manquantes détectées dans les colonnes critiques: {missing_info.to_dict()}")
        
        # --- ÉTAPE 1: Feature Engineering ---
        df_processed['BuildingAge'] = 2016 - df_processed['YearBuilt'].astype(int)
        
        # Protection contre log(0) ou log(négatif)
        df_processed['PropertyGFATotal_log'] = np.log(df_processed['PropertyGFATotal'].clip(lower=1))
        
        df_processed['Distance_to_Center'] = self.haversine_vectorized(
            df_processed['Latitude'].astype(float), 
            df_processed['Longitude'].astype(float), 
            SEATTLE_CENTER_LAT, 
            SEATTLE_CENTER_LON
        )
        df_processed['Is_Downtown'] = (df_processed['Distance_to_Center'] < 2).astype(int)
        
        angle_rad = np.radians(30)
        lat_float = df_processed['Latitude'].astype(float)
        lon_float = df_processed['Longitude'].astype(float)
        df_processed['Rotated_Lat'] = lat_float * np.cos(angle_rad) - lon_float * np.sin(angle_rad)
        df_processed['Rotated_Lon'] = lat_float * np.sin(angle_rad) + lon_float * np.cos(angle_rad)
        
        # --- ÉTAPE 2: Clustering ---
        geo_features = df_processed[['Latitude', 'Longitude']].astype(float).values
        df_processed['Neighborhood_Cluster'] = self.kmeans_geo.predict(geo_features)
        
        surf_features = df_processed[['PropertyGFATotal_log']].astype(float).values
        cluster_ids = self.kmeans_surf.predict(surf_features)
        # FIX: Utiliser une list comprehension au lieu de concaténation NumPy
        df_processed['Surface_Cluster'] = ["surf_group_" + str(cid) for cid in cluster_ids]
        
        # --- ÉTAPE 3: Alignement ---
        missing_cols = set(self.training_columns) - set(df_processed.columns)
        if missing_cols:
            raise ValueError(f"Colonnes manquantes après le feature engineering : {missing_cols}")
        df_processed = df_processed[self.training_columns]
        
        # --- ÉTAPE 4: Encodage ---
        numeric_cols_before_encoding = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        numeric_values_backup = df_processed[numeric_cols_before_encoding].copy()
        print(f"[DEBUG] Colonnes numériques AVANT encodage: {numeric_cols_before_encoding}")
        print(f"[DEBUG] Colonnes à encoder: {self.encoder.cols}")
        
        # Ne convertir en string QUE les colonnes catégorielles
        for col in self.encoder.cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(str).str.strip().str.lower()
        
        df_encoded = self.encoder.transform(df_processed)
        
        # RESTAURER les colonnes numériques après encodage
        print(f"[DEBUG] Restauration des colonnes numériques après encodage...")
        for col in numeric_cols_before_encoding:
            if col in df_encoded.columns and col in numeric_values_backup.columns:
                if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.kind in ['U', 'S']:
                    print(f"[ALERT] Colonne {col} corrompue par l'encodeur (dtype={df_encoded[col].dtype})")
                    df_encoded[col] = numeric_values_backup[col].values
                    print(f"[OK] Colonne {col} restaurée (dtype={df_encoded[col].dtype})")
                else:
                    print(f"[OK] Colonne {col} préservée (dtype={df_encoded[col].dtype})")
        
        # --- ÉTAPE 5: DIAGNOSTIC DÉTAILLÉ ---
        print(f"\n[DEBUG] ========== DIAGNOSTIC APRÈS ENCODAGE ==========")
        print(f"[DEBUG] Shape: {df_encoded.shape}")
        print(f"[DEBUG] Colonnes totales: {len(df_encoded.columns)}")
        
        string_cols = []
        for col in df_encoded.columns:
            dtype = df_encoded[col].dtype
            if dtype == 'object' or dtype.kind == 'U' or dtype.kind == 'S':
                sample_val = df_encoded[col].iloc[0] if len(df_encoded) > 0 else None
                string_cols.append(col)
                print(f"[ALERT] Colonne STRING détectée: {col} | dtype={dtype} | sample={sample_val}")
        
        if string_cols:
            print(f"[ALERT] {len(string_cols)} colonnes STRING détectées: {string_cols}")
        
        # --- ÉTAPE 6: CONVERSION ULTRA-AGRESSIVE ---
        print(f"\n[DEBUG] ========== CONVERSION FORCÉE ==========")
        
        conversion_errors = []
        for col in df_encoded.columns:
            original_dtype = df_encoded[col].dtype
            try:
                df_encoded[col] = df_encoded[col].astype(float)
                print(f"[OK] {col}: {original_dtype} -> float64 (direct)")
            except (ValueError, TypeError) as e:
                try:
                    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
                    df_encoded[col] = df_encoded[col].fillna(0).astype(float)
                    print(f"[OK] {col}: {original_dtype} -> float64 (coerce)")
                except Exception as e2:
                    try:
                        unique_vals = df_encoded[col].unique()
                        print(f"[WARNING] {col}: valeurs uniques = {unique_vals}")
                        mapping = {val: float(i) for i, val in enumerate(unique_vals)}
                        df_encoded[col] = df_encoded[col].map(mapping).fillna(0).astype(float)
                        print(f"[OK] {col}: {original_dtype} -> float64 (mapping)")
                    except Exception as e3:
                        conversion_errors.append((col, str(e3)))
                        print(f"[ERROR] {col}: ÉCHEC TOTAL - {e3}")
                        df_encoded[col] = 0.0
        
        if conversion_errors:
            print(f"\n[ERROR] Erreurs de conversion: {conversion_errors}")
        
        # --- ÉTAPE 7: VÉRIFICATION FINALE ---
        print(f"\n[DEBUG] ========== VÉRIFICATION FINALE ==========")
        non_numeric = []
        for col in df_encoded.columns:
            if df_encoded[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                non_numeric.append((col, df_encoded[col].dtype))
                print(f"[ALERT] Colonne NON-NUMÉRIQUE: {col} | dtype={df_encoded[col].dtype}")
        
        if non_numeric:
            raise ValueError(f"Colonnes non-numériques détectées après conversion: {non_numeric}")
        
        print(f"[OK] Toutes les colonnes sont numériques!")
        print(f"[DEBUG] Résumé des types: {df_encoded.dtypes.value_counts().to_dict()}")
        print(f"[DEBUG] Échantillon:\n{df_encoded.head()}")
            
        return df_encoded
    
    def predict(self, df: pd.DataFrame) -> tuple:
        """Prédiction avec gestion d'erreurs"""
        if not self.loaded:
            raise RuntimeError("Le modèle n'est pas chargé.")
        
        try:
            df_processed = self.preprocess_input(df)
            
            print(f"[DEBUG] Shape avant prédiction: {df_processed.shape}")
            print(f"[DEBUG] Colonnes: {df_processed.columns.tolist()}")
            print(f"[DEBUG] Dtypes: {df_processed.dtypes.value_counts().to_dict()}")
            
            y_pred_log = self.model.predict(df_processed)
            y_pred_real = np.exp(y_pred_log)
            
            return y_pred_real, y_pred_log
        except Exception as e:
            print(f"[ERREUR] Échec de la prédiction: {e}")
            import traceback
            traceback.print_exc()
            raise

model_manager = ModelManager()

# =====================================================
# APPLICATION FASTAPI
# =====================================================

app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

@app.on_event("startup")
async def startup_event():
    """Chargement du modèle au démarrage"""
    model_manager.load_model()

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(building: BuildingInput):
    """Prédiction pour un seul bâtiment"""
    if not model_manager.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model not loaded"
        )
    
    try:
        # CRITIQUE : utiliser by_alias=True pour gérer PropertyGFABuilding(s)
        building_dict = building.dict(by_alias=True)
        print(f"[DEBUG] Données reçues: {building_dict}")
        
        df = pd.DataFrame([building_dict])
        print(f"[DEBUG] DataFrame colonnes: {df.columns.tolist()}")
        print(f"[DEBUG] DataFrame dtypes: {df.dtypes.to_dict()}")
        
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
        error_trace = traceback.format_exc()
        print(f"[ERREUR COMPLÈTE]\n{error_trace}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "trace": error_trace[:1000]
            }
        )

@app.get("/", include_in_schema=False)
async def root():
    """Point d'entrée racine - Serve la page HTML"""
    return FileResponse("index.html")

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Vérification de l'état de l'API"""
    return HealthResponse(
        status="healthy" if model_manager.loaded else "unhealthy",
        api_version=API_VERSION,
        model_loaded=model_manager.loaded,
        model_path=MODEL_PATH,
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/debug/validate", tags=["Debug"])
async def debug_validation(data: dict):
    """Endpoint pour debugger la validation Pydantic"""
    try:
        building = BuildingInput(**data)
        return {
            "status": "valid",
            "parsed_data": building.dict(by_alias=True),
            "dtypes": {k: type(v).__name__ for k, v in building.dict().items()}
        }
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=422,
            content={
                "status": "invalid",
                "error": str(e),
                "trace": traceback.format_exc()
            }
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """Prédiction pour plusieurs bâtiments"""
    if not model_manager.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model not loaded"
        )
    
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
        
        return BatchPredictionResponse(
            predictions=predictions, 
            count=len(predictions), 
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Batch prediction failed: {str(e)}"
        )
    