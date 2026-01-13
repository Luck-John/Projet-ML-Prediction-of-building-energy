import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# =====================================================
# CONSTANTES GÉOGRAPHIQUES (From notebook)
# =====================================================

SEATTLE_CENTER_LAT = 47.6062
SEATTLE_CENTER_LON = -122.3321

# =====================================================
# FONCTIONS UTILITAIRES
# =====================================================

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Calcule la distance de Haversine entre deux points (en km).
    Version vectorisée pour optimiser les performances sur de grands DataFrames.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


# =====================================================
# FEATURE ENGINEERING - GEOGRAPHIC FEATURES
# =====================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline matching notebook logic exactly.
    Assumes df already has:
    - SiteEnergyUse_log, PropertyGFATotal_log (from preprocessing)
    - Latitude, Longitude (raw geographic coordinates)
    
    Returns: DataFrame with exactly 24 columns for final export
    """
    df = df.copy()
    
    print(">>> Creating geographic features...")
    
    # 1. Distance to Seattle Center (Haversine formula)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        df['Distance_to_Center'] = haversine_vectorized(
            df['Latitude'], df['Longitude'],
            SEATTLE_CENTER_LAT, SEATTLE_CENTER_LON
        )
        print(f"   [OK] Distance_to_Center created")
    
    # 2. Neighborhood Clustering (KMeans with 10 clusters)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        df['Neighborhood_Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])
        print(f"   [OK] Neighborhood_Cluster created (10 clusters)")
    
    # 3. Is_Downtown binary (within 2 km of center)
    if 'Distance_to_Center' in df.columns:
        df['Is_Downtown'] = (df['Distance_to_Center'] < 2).astype(int)
        print(f"   [OK] Is_Downtown created")
    
    # 4. Rotated Coordinates (30 degree rotation)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        angle_rad = np.radians(30)
        df['Rotated_Lat'] = df['Latitude'] * np.cos(angle_rad) - df['Longitude'] * np.sin(angle_rad)
        df['Rotated_Lon'] = df['Latitude'] * np.sin(angle_rad) + df['Longitude'] * np.cos(angle_rad)
        print(f"   [OK] Rotated_Lat and Rotated_Lon created (30 degree rotation)")
    
    # 5. Surface Clustering (KMeans with 2 clusters on log-surface)
    if 'PropertyGFATotal_log' in df.columns:
        kmeans_surf = KMeans(n_clusters=2, random_state=42, n_init=10)
        df['Surface_Cluster'] = kmeans_surf.fit_predict(df[['PropertyGFATotal_log']])
        df['Surface_Cluster'] = "Surf_Group_" + df['Surface_Cluster'].astype(str)
        print(f"   [OK] Surface_Cluster created (2 groups)")
    
    # 6. Select ONLY the 24 final columns (matching notebook export)
    final_columns = [
        'BuildingType', 'PrimaryPropertyType', 'ZipCode', 'CouncilDistrictCode',
        'Neighborhood', 'NumberofBuildings', 'NumberofFloors', 'PropertyGFATotal',
        'PropertyGFAParking', 'PropertyGFABuilding(s)', 'ListOfAllPropertyUseTypes',
        'LargestPropertyUseType', 'LargestPropertyUseTypeGFA', 'ENERGYSTARScore',
        'SiteEnergyUse(kBtu)', 'BuildingAge', 'SiteEnergyUse_log', 'PropertyGFATotal_log',
        'Distance_to_Center', 'Neighborhood_Cluster', 'Is_Downtown', 'Rotated_Lat', 'Rotated_Lon',
        'Surface_Cluster'
    ]
    
    # Keep only columns that exist
    df = df[[c for c in final_columns if c in df.columns]]
    
    print(f"   [OK] Selected {df.shape[1]} final columns")
    print(f"   [OK] Final shape: {df.shape}")
    return df


def prepare_for_training(df: pd.DataFrame) -> tuple:
    """
    Final preparation for training.
    Returns: (X, y) where X has no target, y is the log-transformed target
    """
    df = df.copy()
    
    target_col = 'SiteEnergyUse_log'
    
    # Extract target
    y = df[target_col]
    
    # Create features (drop only the target column and original non-log target)
    X = df.drop(columns=[target_col, 'SiteEnergyUse(kBtu)'], errors='ignore')
    
    return X, y
