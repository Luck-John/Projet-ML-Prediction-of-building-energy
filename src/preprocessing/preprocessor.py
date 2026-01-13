import os
import json
import hashlib
from datetime import datetime

import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional
from sklearn.preprocessing import MultiLabelBinarizer
import category_encoders as ce

# =====================================================
# CONSTANTES
# =====================================================

DATA_URL = "https://raw.githubusercontent.com/MouslyDiaw/handson-machine-learning/refs/heads/master/data/2016_Building_Energy_Benchmarking.csv"

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

STD_CAT_COLS = ['PrimaryPropertyType', 'BuildingType', 'Neighborhood', 'LargestPropertyUseType']
TARGET_COL = 'SiteEnergyUse(kBtu)'
SPARSE_THRESHOLD = 0.4

# Outlier thresholds (from notebook analysis)
HIGH_CONSUMPTION_THRESHOLD = 2.0 * 10**8  # 200,000,000 kBtu
HIGH_SURFACE_THRESHOLD = 3.0 * 10**6      # 3,000,000 sqft


# =====================================================
# ÉTAPE 1 : NETTOYAGE INITIAL
# =====================================================

def load_data(path_or_url: str = DATA_URL) -> pd.DataFrame:
    """Load raw data from CSV"""
    df = pd.read_csv(path_or_url)
    return df


def filter_non_residential(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to non-residential buildings only"""
    return df[~df['BuildingType'].isin(RESIDENTIAL_TYPES)].copy()


def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop useless or data leakage columns"""
    cols = [c for c in COLS_TO_DROP if c in df.columns]
    return df.drop(columns=cols)


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the target variable"""
    df = df[df[TARGET_COL] > 0]
    df = df.dropna(subset=[TARGET_COL])
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate buildings by OSEBuildingID"""
    if 'OSEBuildingID' not in df.columns:
        return df
    return df.drop_duplicates(subset=['OSEBuildingID'], keep='first')


def drop_sparse_columns(df: pd.DataFrame, threshold: float = SPARSE_THRESHOLD) -> pd.DataFrame:
    """Drop columns with >threshold missing values"""
    missing_ratio = df.isnull().mean()
    sparse_cols = missing_ratio[missing_ratio > threshold].index.tolist()
    return df.drop(columns=sparse_cols)


def impute_energy_star(df: pd.DataFrame) -> pd.DataFrame:
    """Impute ENERGYSTARScore with median by property type"""
    if 'ENERGYSTARScore' in df.columns:
        df['ENERGYSTARScore'] = df['ENERGYSTARScore'].fillna(
            df.groupby('PrimaryPropertyType')['ENERGYSTARScore'].transform('median')
        )
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extreme outliers based on consumption and surface"""
    df = df[df[TARGET_COL] < HIGH_CONSUMPTION_THRESHOLD].copy()
    df = df[df['PropertyGFATotal'] < HIGH_SURFACE_THRESHOLD].copy()
    return df


def final_dropna(df: pd.DataFrame) -> pd.DataFrame:
    """Final removal of rows with missing values"""
    return df.dropna()


# =====================================================
# ÉTAPE 2 : TRANSFORMATIONS LOGARITHMIQUES
# =====================================================

def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log transformations to target and surface"""
    df['SiteEnergyUse_log'] = np.log(df[TARGET_COL])
    df['PropertyGFATotal_log'] = np.log(df['PropertyGFATotal'])
    df['BuildingAge'] = 2016 - df['YearBuilt']
    return df


# =====================================================
# ÉTAPE 3 : ENCODAGE CATÉGORIQUE (TARGET ENCODING)
# =====================================================

def encode_categories(df: pd.DataFrame, target_col: str = 'SiteEnergyUse_log') -> Tuple[pd.DataFrame, ce.TargetEncoder]:
    """
    Apply Target Encoding to categorical variables.
    Returns: (df_encoded, encoder)
    """
    std_cat_cols = [c for c in STD_CAT_COLS if c in df.columns]
    
    if not std_cat_cols:
        return df, None
    
    # Lowercase categorical columns
    for col in std_cat_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
    
    # Target Encoding
    encoder = ce.TargetEncoder(cols=std_cat_cols)
    df_encoded = encoder.fit_transform(df[std_cat_cols], df[target_col])
    
    # Replace original columns with encoded versions
    df[std_cat_cols] = df_encoded
    
    return df, encoder


def create_mlb_features(df: pd.DataFrame, list_col: str = 'ListOfAllPropertyUseTypes') -> pd.DataFrame:
    """
    Create multi-label binarized features from comma-separated list column.
    Returns DataFrame of MLB features (may be empty if column missing).
    """
    if list_col not in df.columns:
        return pd.DataFrame(index=df.index)

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
# FONCTION PRINCIPALE
# =====================================================

def preprocess_data(path_or_url: str = DATA_URL) -> pd.DataFrame:
    """Complete preprocessing pipeline matching notebook logic exactly"""
    
    # Step 1: Load data
    df = load_data(path_or_url)
    print(f"[OK] Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    
    # Step 2: Filter non-residential
    df = filter_non_residential(df)
    print(f"[OK] Filtered to non-residential: {df.shape[0]} rows")
    
    # Step 3: Drop unusable columns
    df = drop_unusable_columns(df)
    
    # Step 4: Remove duplicates
    df = remove_duplicates(df)
    
    # Step 5: Clean target
    df = clean_target(df)
    print(f"[OK] Cleaned target: {df.shape[0]} rows")
    
    # Step 6: Drop sparse columns
    df = drop_sparse_columns(df)
    
    # Step 7: Impute ENERGY STAR
    df = impute_energy_star(df)
    
    # Step 8: Final dropna
    df = final_dropna(df)
    print(f"[OK] Handled missing values: {df.shape[0]} rows, {df.shape[1]} cols")
    
    # Step 9: Remove outliers
    df = remove_outliers(df)
    print(f"[OK] Removed outliers: {df.shape[0]} rows")
    
    # Step 10: Apply log transforms
    df = apply_log_transforms(df)
    print(f"[OK] Applied log transforms")
    
    print(f"[OK] Preprocessing complete: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def preprocess_data_production(path_or_url: str = DATA_URL) -> pd.DataFrame:
    """
    Production preprocessing pipeline WITHOUT target variable.
    
    For making predictions on new data where the target is unknown.
    This function does NOT clean the target or remove target-based outliers.
    """
    
    # Step 1: Load data
    df = load_data(path_or_url)
    print(f"[OK] Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    
    # Step 2: Filter non-residential
    df = filter_non_residential(df)
    print(f"[OK] Filtered to non-residential: {df.shape[0]} rows")
    
    # Step 3: Drop unusable columns (but DO NOT drop target - keep it for reference)
    # In production, target column will be missing anyway
    df = drop_unusable_columns(df)
    
    # Step 4: Remove duplicates
    df = remove_duplicates(df)
    
    # NOTE: Skip clean_target and remove_outliers (require target variable)
    
    # Step 5: Drop sparse columns
    df = drop_sparse_columns(df)
    
    # Step 6: Impute ENERGY STAR
    df = impute_energy_star(df)
    
    # Step 7: Final dropna
    df = final_dropna(df)
    print(f"[OK] Handled missing values: {df.shape[0]} rows, {df.shape[1]} cols")
    
    # Step 8: Apply log transforms (without target cleaning)
    df['BuildingAge'] = 2016 - df['YearBuilt']
    if 'PropertyGFATotal' in df.columns and (df['PropertyGFATotal'] > 0).all():
        df['PropertyGFATotal_log'] = np.log(df['PropertyGFATotal'])
    print(f"[OK] Applied log transforms")
    
    print(f"[OK] Production preprocessing complete: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


# =====================================================
# VERSIONNING & EXPORT
# =====================================================

def _compute_df_hash(df: pd.DataFrame) -> str:
    """Compute SHA256 hash of dataframe content"""
    content = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(content.tobytes()).hexdigest()


def save_processed_df(df: pd.DataFrame, 
                     output_path: str = "data/processed/seattle_energy_cleaned_final.csv",
                     version_path: str = "artifacts/data_version.json",
                     force: bool = False) -> None:
    """
    Save processed dataframe with versioning metadata.
    Creates version JSON with SHA256, timestamp, shape.
    """
    # Ensure directories exist
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(version_path) or '.', exist_ok=True)
    
    # Remove old files if force=True (prevent Windows locking)
    if force:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception as e:
                print(f"Warning: Could not remove {output_path}: {e}")
        if os.path.exists(version_path):
            try:
                os.remove(version_path)
            except Exception as e:
                print(f"Warning: Could not remove {version_path}: {e}")
    
    # Save CSV
    df.to_csv(output_path, index=False)
    
    # Create version metadata
    data_version = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sha256": _compute_df_hash(df),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns)
    }
    
    # Save version JSON
    with open(version_path, 'w') as f:
        json.dump(data_version, f, indent=2)
    
    print(f"[OK] Saved: {output_path} ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(f"[OK] Saved: {version_path}")
