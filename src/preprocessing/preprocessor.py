import pandas as pd
import numpy as np
from typing import Union

# =====================================================
# CONSTANTES
# =====================================================

RESIDENTIAL_TYPES = [
    'Multifamily LR (1-4)',
    'Multifamily MR (5-9)',
    'Multifamily HR (10+)'
]

COLS_TO_DROP = [
    'OSEBuildingID',
    'DataYear',
    'PropertyName',
    'City',
    'State',
    'TaxParcelIdentificationNumber',
    'SiteEUI(kBtu/sf)',
    'SiteEUIWN(kBtu/sf)',
    'SourceEUI(kBtu/sf)',
    'SourceEUIWN(kBtu/sf)',
    'SiteEnergyUseWN(kBtu)',
    'DefaultData',
    'Comments',
    'ComplianceStatus',
    'Outlier',
    'TotalGHGEmissions',
    'GHGEmissionsIntensity',
    'Address',
    'SteamUse(kBtu)',
    'Electricity(kBtu)',
    'Electricity(kWh)',
    'NaturalGas(therms)',
    'NaturalGas(kBtu)'
]

TARGET_COL = 'SiteEnergyUse(kBtu)'

SPARSE_THRESHOLD = 0.4  # 40%

# Outlier thresholds used in the notebook
HIGH_CONSUMPTION_THRESHOLD = 2.0 * 10**8  # 200,000,000 kBtu
HIGH_SURFACE_THRESHOLD = 3.0 * 10**6      # 3,000,000 sqft


# =====================================================
# FONCTIONS DE PREPROCESSING
# =====================================================


def load_data(path_or_url: str) -> pd.DataFrame:
    """Charge les données brutes"""
    df = pd.read_csv(path_or_url)
    return df


def filter_non_residential(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les bâtiments résidentiels"""
    return df[~df['BuildingType'].isin(RESIDENTIAL_TYPES)].copy()


def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes inutiles ou fuyantes"""
    cols = [c for c in COLS_TO_DROP if c in df.columns]
    return df.drop(columns=cols)


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage de la variable cible"""
    df = df[df[TARGET_COL] > 0]
    df = df.dropna(subset=[TARGET_COL])
    return df


def drop_sparse_columns(df: pd.DataFrame, threshold: float = SPARSE_THRESHOLD) -> pd.DataFrame:
    """Supprime les colonnes avec trop de valeurs manquantes"""
    missing_ratio = df.isnull().mean()
    sparse_cols = missing_ratio[missing_ratio > threshold].index.tolist()
    return df.drop(columns=sparse_cols)


def impute_energy_star(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation de ENERGY STAR Score
    Médiane par type de propriété (logique métier)
    """
    if 'ENERGYSTARScore' in df.columns:
        df['ENERGYSTARScore'] = df['ENERGYSTARScore'].fillna(
            df.groupby('PrimaryPropertyType')['ENERGYSTARScore'].transform('median')
        )
    return df


def final_dropna(df: pd.DataFrame) -> pd.DataFrame:
    """Suppression finale des lignes encore incomplètes"""
    return df.dropna()


def _lowercase_cats(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def _create_mlb_features(df: pd.DataFrame, list_col: str = 'ListOfAllPropertyUseTypes') -> pd.DataFrame:
    """Explode a comma-separated list column into multi-hot columns using MultiLabelBinarizer.
    Returns a DataFrame of mlb features (may be empty if column missing).
    """
    if list_col not in df.columns:
        return pd.DataFrame(index=df.index)

    from sklearn.preprocessing import MultiLabelBinarizer

    temp_list = df[list_col].fillna('').astype(str).apply(
        lambda x: [i.strip().lower() for i in x.split(',')] if x != '' else []
    )

    mlb = MultiLabelBinarizer()
    if len(temp_list) == 0:
        return pd.DataFrame(index=df.index)

    mat = mlb.fit_transform(temp_list)
    cols = [f"use_{c}" for c in mlb.classes_]
    df_mlb = pd.DataFrame(mat, columns=cols, index=df.index)
    return df_mlb


# =====================================================
# API PUBLICA
# =====================================================


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Applique la suite de transformations sur un DataFrame déjà chargé."""
    df = filter_non_residential(df)
    df = drop_unusable_columns(df)
    df = clean_target(df)
    df = drop_sparse_columns(df)
    df = impute_energy_star(df)

    # Drop remaining NaNs
    df = final_dropna(df)

    # Notebook-specific: remove extreme outliers (consumption & surface)
    if 'SiteEnergyUse(kBtu)' in df.columns:
        df = df[df['SiteEnergyUse(kBtu)'] < HIGH_CONSUMPTION_THRESHOLD].copy()
    if 'PropertyGFATotal' in df.columns:
        df = df[df['PropertyGFATotal'] < HIGH_SURFACE_THRESHOLD].copy()

    # Normalize categorical text to lower/strip for key categorical columns used in notebook
    std_cat_cols = ['PrimaryPropertyType', 'BuildingType', 'Neighborhood', 'LargestPropertyUseType']
    df = _lowercase_cats(df, std_cat_cols)

    # Create MLB features for ListOfAllPropertyUseTypes and concat to dataframe
    df_mlb = _create_mlb_features(df, list_col='ListOfAllPropertyUseTypes')
    if not df_mlb.empty:
        df = pd.concat([df, df_mlb], axis=1)

    # Transformations log (Notebook 11 - ligne 1143-1144)
    if 'SiteEnergyUse(kBtu)' in df.columns:
        df['SiteEnergyUse_log'] = np.log(df['SiteEnergyUse(kBtu)'])
    if 'PropertyGFATotal' in df.columns:
        df['PropertyGFATotal_log'] = np.log(df['PropertyGFATotal'])

    return df


def preprocess_data(path_or_url: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Pipeline complet de preprocessing.

    Accepte soit un chemin/URL, soit un DataFrame déjà chargé.
    """
    if isinstance(path_or_url, pd.DataFrame):
        return preprocess_df(path_or_url.copy())

    df = load_data(path_or_url)
    return preprocess_df(df)
