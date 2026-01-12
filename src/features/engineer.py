import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# =====================================================
# CONSTANTES GÉOGRAPHIQUES
# =====================================================

SEATTLE_CENTER_LAT = 47.6062
SEATTLE_CENTER_LON = -122.3321

# =====================================================
# FONCTIONS UTILITAIRES
# =====================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Distance de Haversine vectorisée (en km)"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


# =====================================================
# FEATURE ENGINEERING PRINCIPAL
# =====================================================

def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transformations log"""
    df = df.copy()
    df['SiteEnergyUse_log'] = np.log(df['SiteEnergyUse(kBtu)'])
    df['PropertyGFATotal_log'] = np.log(df['PropertyGFATotal'])
    return df


def add_building_age(df: pd.DataFrame, reference_year: int = 2016) -> pd.DataFrame:
    """Âge du bâtiment"""
    df = df.copy()
    if 'YearBuilt' in df.columns:
        df['BuildingAge'] = reference_year - df['YearBuilt']
        df.drop(columns=['YearBuilt'], inplace=True)
    return df


def add_geographical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features géographiques"""
    df = df.copy()

    # Si pas de coordonnées, on skippe les features géographiques
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        return df

    # Distance au centre
    df['Distance_to_Center'] = haversine_distance(
        df['Latitude'], df['Longitude'],
        SEATTLE_CENTER_LAT, SEATTLE_CENTER_LON
    )

    # Cluster spatial
    kmeans_geo = KMeans(n_clusters=10, random_state=42, n_init=10)
    df['Neighborhood_Cluster'] = kmeans_geo.fit_predict(
        df[['Latitude', 'Longitude']]
    )

    # Centre-ville
    df['Is_Downtown'] = (df['Distance_to_Center'] < 2).astype(int)

    # Coordonnées rotatées
    angle = np.radians(30)
    df['Rotated_Lat'] = df['Latitude'] * np.cos(angle) - df['Longitude'] * np.sin(angle)
    df['Rotated_Lon'] = df['Latitude'] * np.sin(angle) + df['Longitude'] * np.cos(angle)

    # Nettoyage
    df.drop(columns=['Latitude', 'Longitude'], inplace=True)

    return df


def add_surface_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster de surface"""
    df = df.copy()

    if 'PropertyGFATotal_log' not in df.columns:
        return df

    kmeans_surf = KMeans(n_clusters=2, random_state=42, n_init=10)
    df['Surface_Cluster'] = kmeans_surf.fit_predict(
        df[['PropertyGFATotal_log']]
    )

    df['Surface_Cluster'] = "Surf_Group_" + df['Surface_Cluster'].astype(str)
    return df


# =====================================================
# PIPELINE COMPLET
# =====================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet de feature engineering
    """
    df = add_log_features(df)
    df = add_building_age(df)
    df = add_geographical_features(df)
    df = add_surface_cluster(df)

    return df
