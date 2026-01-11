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

model, encoder, best_params = load_model()

# ============================================================================
# Header
# ============================================================================

st.title("🏢 Building Energy Prediction Dashboard")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Type", "StackingRegressor")
with col2:
    st.metric("Test MAPE", "0.4201 (21%)")
with col3:
    st.metric("Test R²", "0.527")

# ============================================================================
# Sidebar - Navigation
# ============================================================================

st.sidebar.header("📋 Navigation")
page = st.sidebar.radio(
    "Choisir une page:",
    ["🔮 Prédiction", "📊 Données", "📈 Modèle", "ℹ️ À Propos"]
)

# ============================================================================
# PAGE 1: PRÉDICTION
# ============================================================================

if page == "🔮 Prédiction":
    st.header("Prédire la Consommation Énergétique")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Entrées")
        property_gfa = st.number_input("Surface Totale (sqft)", min_value=0, value=50000)
        year_built = st.number_input("Année de Construction", min_value=1900, max_value=2024, value=2005)
        energy_star = st.number_input("ENERGYSTARScore", min_value=0, max_value=100, value=75)
        
        property_type = st.selectbox(
            "Type de Propriété",
            ["Office", "Retail", "Hotel", "Warehouse", "Medical Office", "Data Center", "Other"]
        )
    
    with col2:
        st.subheader("🎯 Prédiction")
        
        if st.button("🚀 Prédire", use_container_width=True):
            try:
                # Créer DataFrame
                X = pd.DataFrame({
                    'PropertyGFATotal': [property_gfa],
                    'YearBuilt': [year_built],
                    'ENERGYSTARScore': [energy_star],
                    'PrimaryPropertyType': [property_type]
                })
                
                # Encoder
                if encoder:
                    X_encoded = encoder.transform(X)
                else:
                    X_encoded = X
                
                # Prédire
                pred_log = model.predict(X_encoded)[0]
                pred_real = np.exp(pred_log)
                
                # Afficher résultat
                st.success("✅ Prédiction Réussie!")
                
                st.metric("Consommation Énergétique", f"{pred_real:,.0f} kBtu")
                
                # Jauge de consommation
                if pred_real < 1e6:
                    severity = "🟢 Faible"
                elif pred_real < 5e6:
                    severity = "🟡 Moyen"
                else:
                    severity = "🔴 Élevé"
                
                st.info(f"Niveau: {severity}")
                
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

# ============================================================================
# PAGE 2: DONNÉES
# ============================================================================

elif page == "📊 Données":
    st.header("📊 Informations sur les Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Dataset Statistics")
        st.info("""
        - **Nombre de bâtiments:** 1,553
        - **Année:** 2016
        - **Région:** Seattle, Washington
        - **Type:** Non-résidentiel uniquement
        - **Consommation moyenne:** 2.4M kBtu
        - **Consommation min:** 50K kBtu
        - **Consommation max:** 250M kBtu
        """)
    
    with col2:
        st.subheader("🧹 Nettoyage Appliqué")
        st.info("""
        ✅ Filtrage: Bâtiments non-résidentiels
        ✅ Valeurs aberrantes supprimées
        ✅ Log-transformation: SiteEnergyUse_log
        ✅ Target Encoding: Variables catégorielles
        ✅ Feature Engineering: Distance, Clusters, Age
        """)

# ============================================================================
# PAGE 3: MODÈLE
# ============================================================================

elif page == "📈 Modèle":
    st.header("📈 Architecture du Modèle")
    
    st.subheader("🏗️ StackingRegressor")
    st.markdown("""
    **Base Learners (Grid Search):**
    - ExtraTrees: n_estimators=500, max_depth=10
    - XGBoost: n_estimators=300, learning_rate=0.05, max_depth=3
    - LightGBM: n_estimators=100, learning_rate=0.05, num_leaves=50
    - HistGradientBoosting: learning_rate=0.05, max_iter=200
    
    **Meta-Learner:**
    - LinearSVR(C=10, dual='auto', max_iter=10000)
    """)
    
    st.subheader("📊 Performances")
    metrics = {
        "MAPE (Real)": "0.4201 (21% error)",
        "R² (Real)": "0.527",
        "RMSE (Real)": "7,877,872 kBtu",
        "MAE (Real)": "2,396,297 kBtu"
    }
    
    cols = st.columns(len(metrics))
    for i, (metric_name, metric_value) in enumerate(metrics.items()):
        with cols[i]:
            st.metric(metric_name, metric_value)

# ============================================================================
# PAGE 4: À PROPOS
# ============================================================================

elif page == "ℹ️ À Propos":
    st.header("ℹ️ À Propos du Projet")
    
    st.markdown("""
    ### 🎯 Objectif
    Prédire la consommation totale d'énergie des bâtiments non-résidentiels de Seattle
    et évaluer la pertinence du score ENERGYSTARScore.
    
    ### 📚 Dataset
    - **Source:** 2016 Building Energy Benchmarking (Seattle)
    - **Bâtiments:** 1,553 non-résidentiels
    - **Variables:** 30+ (surface, année, type, scores énergétiques, etc.)
    
    ### 🔧 Stack Technique
    - **ML:** scikit-learn, XGBoost, LightGBM, category_encoders
    - **API:** FastAPI
    - **Dashboard:** Streamlit
    - **Tracking:** MLflow
    - **CI/CD:** GitHub Actions
    
    ### 👥 Collaborateurs
    - [Malick Sene](https://github.com/malickseneisep2)
    - [Ameth Faye](https://github.com/ameth08faye)
    - [Hilda Edima](https://github.com/HildaEDIMA)
    - [Albert Zinaba](https://github.com/ZINABA-Albert)
    
    ### 📁 Repository
    https://github.com/Luck-John/Projet-ML-Prediction-of-building-energy
    """)

# ============================================================================
# Footer
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit & ML")
