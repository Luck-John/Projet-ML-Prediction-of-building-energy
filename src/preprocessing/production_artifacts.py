"""
Utilities to load encoder and KMeans models for production inference.
Simple helper functions for the API team.
"""

import os
import joblib
from typing import Optional


def load_encoder_from_model(model_path: str = "artifacts/model.joblib"):
    """
    Load the TargetEncoder from the saved model dictionary.
    
    Args:
        model_path: Path to artifacts/model.joblib
        
    Returns:
        encoder object or None
        
    Example:
        >>> encoder = load_encoder_from_model()
        >>> X_encoded = encoder.transform(X[encoder.cols])
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_dict = joblib.load(model_path)
    encoder = model_dict.get('encoder')
    
    if encoder is None:
        raise ValueError("No encoder found in model dictionary")
    
    print(f"[OK] Encoder loaded from {model_path}")
    print(f"     Columns: {encoder.cols}")
    print(f"     Handle unknown: {getattr(encoder, 'handle_unknown', 'not specified')}")
    
    return encoder


def load_kmeans_neighborhood(path: str = "artifacts/kmeans_neighborhood.joblib"):
    """
    Load pre-trained KMeans for neighborhood clustering.
    
    Args:
        path: Path to kmeans_neighborhood.joblib
        
    Returns:
        KMeans model
        
    Example:
        >>> kmeans = load_kmeans_neighborhood()
        >>> clusters = kmeans.predict(df[['Latitude', 'Longitude']])
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"KMeans neighborhood model not found: {path}")
    
    kmeans = joblib.load(path)
    print(f"[OK] KMeans neighborhood loaded: {kmeans.n_clusters} clusters")
    
    return kmeans


def load_kmeans_surface(path: str = "artifacts/kmeans_surface.joblib"):
    """
    Load pre-trained KMeans for surface clustering.
    
    Args:
        path: Path to kmeans_surface.joblib
        
    Returns:
        KMeans model
        
    Example:
        >>> kmeans = load_kmeans_surface()
        >>> clusters = kmeans.predict(df[['PropertyGFATotal_log']])
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"KMeans surface model not found: {path}")
    
    kmeans = joblib.load(path)
    print(f"[OK] KMeans surface loaded: {kmeans.n_clusters} clusters")
    
    return kmeans


def load_all_artifacts(
    model_path: str = "artifacts/model.joblib",
    kmeans_neighborhood_path: str = "artifacts/kmeans_neighborhood.joblib",
    kmeans_surface_path: str = "artifacts/kmeans_surface.joblib"
):
    """
    Load all production artifacts at once.
    
    Args:
        model_path: Path to model.joblib
        kmeans_neighborhood_path: Path to kmeans_neighborhood.joblib
        kmeans_surface_path: Path to kmeans_surface.joblib
        
    Returns:
        dict with keys: 'model', 'encoder', 'kmeans_neighborhood', 'kmeans_surface'
        
    Example:
        >>> artifacts = load_all_artifacts()
        >>> encoder = artifacts['encoder']
        >>> kmeans_neighborhood = artifacts['kmeans_neighborhood']
    """
    print("[OK] Loading all production artifacts...")
    
    # Load main model dict
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_dict = joblib.load(model_path)
    
    # Load encoder
    encoder = model_dict.get('encoder')
    if encoder is None:
        print("[WARNING] No encoder found in model dictionary")
    
    # Load KMeans models
    kmeans_neighborhood = None
    kmeans_surface = None
    
    if os.path.exists(kmeans_neighborhood_path):
        kmeans_neighborhood = joblib.load(kmeans_neighborhood_path)
        print(f"[OK] KMeans neighborhood loaded: {kmeans_neighborhood.n_clusters} clusters")
    else:
        print(f"[WARNING] KMeans neighborhood not found: {kmeans_neighborhood_path}")
    
    if os.path.exists(kmeans_surface_path):
        kmeans_surface = joblib.load(kmeans_surface_path)
        print(f"[OK] KMeans surface loaded: {kmeans_surface.n_clusters} clusters")
    else:
        print(f"[WARNING] KMeans surface not found: {kmeans_surface_path}")
    
    return {
        'model': model_dict['model'],
        'encoder': encoder,
        'kmeans_neighborhood': kmeans_neighborhood,
        'kmeans_surface': kmeans_surface,
        'best_params': model_dict.get('best_params', {}),
        'target_col': model_dict.get('target_col', 'SiteEnergyUse_log')
    }
