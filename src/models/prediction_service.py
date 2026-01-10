"""
Service de Prédiction Générique
Logique centralisée pour prédictions, indépendante de l'interface (API, Dashboard, CLI, etc.)
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.preprocessing.preprocessor import preprocess_df
from src.features.engineer import engineer_features


class PredictionService:
    """Service centralisé pour charger modèles et faire des prédictions"""
    
    def __init__(self, use_energy_star: bool = True):
        """
        Initialise le service avec le meilleur modèle approprié
        
        Args:
            use_energy_star: Si True, utilise le modèle entraîné avec ENERGY_STAR
                           Si False, utilise le modèle sans ENERGY_STAR
        """
        self.use_energy_star = use_energy_star
        self.artifacts_path = Path(__file__).parent.parent.parent / "artifacts"
        
        # Charger le modèle et l'encodeur
        model_name = "best_model_with_score" if use_energy_star else "best_model_no_score"
        model_path = self.artifacts_path / f"{model_name}.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        artifact = joblib.load(model_path)
        self.model = artifact['model']
        self.encoder = artifact['encoder']
        self.scaler = artifact.get('scaler', None)
        
    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prédit pour un seul enregistrement (format dictionnaire)
        
        Args:
            record: Dict avec les colonnes du bâtiment
                   Exemple: {'PrimaryPropertyType': 'Office', 'PropertyGFATotal': 50000, ...}
        
        Returns:
            Dict avec prédiction (réelle et log)
        """
        # Conversion en DataFrame
        df = pd.DataFrame([record])
        
        # Prétraitement
        df = preprocess_df(df, use_energy_star=self.use_energy_star)
        
        # Feature engineering
        df = engineer_features(df)
        
        # Adaptation au scénario ENERGY_STAR
        if not self.use_energy_star and 'ENERGYSTARScore' in df.columns:
            df = df.drop(columns=['ENERGYSTARScore'])
        
        # Encodage (variables catégorielles)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            df = self.encoder.transform(df)
        
        # Scaling si disponible
        if self.scaler is not None:
            df = self.scaler.transform(df)
        
        # Prédiction (en log)
        y_pred_log = self.model.predict(df)[0]
        
        # Conversion en unités réelles
        y_pred_real = np.exp(y_pred_log)
        
        return {
            'prediction_kbtu': float(y_pred_real),
            'prediction_log': float(y_pred_log),
            'model_type': 'With ENERGY_STAR' if self.use_energy_star else 'Without ENERGY_STAR',
            'unit': 'kBtu'
        }
    
    def predict_batch(self, records: list) -> pd.DataFrame:
        """
        Prédit pour plusieurs enregistrements
        
        Args:
            records: List of dicts
        
        Returns:
            DataFrame avec prédictions
        """
        results = []
        for record in records:
            result = self.predict_single(record)
            result['input'] = record
            results.append(result)
        return pd.DataFrame(results)
    
    @staticmethod
    def get_required_columns() -> Dict[str, str]:
        """Retourne la liste des colonnes requises avec leurs types"""
        return {
            'PrimaryPropertyType': 'str (ex: Office, Retail, etc.)',
            'BuildingType': 'str (ex: Commercial, Other)',
            'PropertyGFATotal': 'float (pieds carrés)',
            'ENERGYSTARScore': 'float (optionnel si modèle sans score)',
            'YearBuilt': 'int (année)',
            'Latitude': 'float (degré)',
            'Longitude': 'float (degré)',
            'Neighborhood': 'str (quartier)',
            'LargestPropertyUseType': 'str (type d\'usage)',
            'ListOfAllPropertyUseTypes': 'str (comma-separated, ex: Office, Retail)',
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne des infos sur le modèle chargé"""
        return {
            'model_name': 'best_model_with_score' if self.use_energy_star else 'best_model_no_score',
            'use_energy_star': self.use_energy_star,
            'expected_mape': 0.4041 if self.use_energy_star else 0.4950,
            'expected_r2': 0.5248 if self.use_energy_star else 0.5136,
            'model_type': 'ExtraTreesRegressor',
        }


# --- EXEMPLES D'UTILISATION ---

if __name__ == '__main__':
    # Exemple 1: Prédiction simple
    print("=" * 60)
    print("EXEMPLE 1: Prédiction Simple")
    print("=" * 60)
    
    service = PredictionService(use_energy_star=True)
    
    record = {
        'PrimaryPropertyType': 'Office',
        'BuildingType': 'Commercial',
        'PropertyGFATotal': 100000.0,
        'ENERGYSTARScore': 75.0,
        'YearBuilt': 2005,
        'Latitude': 47.6,
        'Longitude': -122.3,
        'Neighborhood': 'Downtown Seattle',
        'LargestPropertyUseType': 'Office',
        'ListOfAllPropertyUseTypes': 'Office'
    }
    
    result = service.predict_single(record)
    print(f"Prédiction: {result['prediction_kbtu']:.2f} kBtu")
    print(f"Type: {result['model_type']}")
    
    # Exemple 2: Info du modèle
    print("\n" + "=" * 60)
    print("INFO MODÈLE")
    print("=" * 60)
    info = service.get_model_info()
    for key, val in info.items():
        print(f"{key}: {val}")
