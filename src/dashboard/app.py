import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# DASHBOARD STREAMLIT - TEMPLATE PRÊT À UTILISER
# ============================================================================
# Fichier: src/dashboard/app.py
# À créer et utiliser pour un dashboard interactif

"""
DÉMARRER LE DASHBOARD:
    streamlit run src/dashboard/app.py

PUIS:
    Ouvre http://localhost:8501 dans ton navigateur
"""

# ============================================================================
# Configuration Streamlit
# ============================================================================

st.set_page_config(
    page_title="Building Energy Prediction",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Charger le modèle
# ============================================================================

@st.cache_resource
def load_model():
    """Charger le modèle avec cache"""
    MODEL_PATH = Path("artifacts/model.joblib")
    if not MODEL_PATH.exists():
        st.error(f"❌ Model not found: {MODEL_PATH}")
        st.stop()
    
    model_dict = joblib.load(MODEL_PATH)
    return model_dict['model'], model_dict['encoder'], model_dict['best_params']
