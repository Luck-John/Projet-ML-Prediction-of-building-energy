# 🏢 Building Energy Prediction - Seattle

> **Prédire la consommation énergétique des bâtiments non-résidentiels de Seattle avec Machine Learning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green)](https://github.com/Luck-John/Projet-ML-Prediction-of-building-energy/actions)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009485.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)

---

## 🎯 Objectif du Projet

Développer un **modèle de Machine Learning** pour:
- ✅ Prédire la **consommation énergétique** (SiteEnergyUse en kBtu)
- ✅ Évaluer la pertinence du **ENERGYSTARScore** dans la prédiction
- ✅ Fournir une **API REST** et un **dashboard interactif**
- ✅ Implémenter un **pipeline MLOps** robuste avec CI/CD

**Dataset:** 2016 Building Energy Benchmarking (Seattle) - 1,553 bâtiments non-résidentiels

---

## 👥 Équipe du Projet

| Collaborateur | GitHub |
|---|---|---|
| Malick SENE | [@malickseneisep2](https://github.com/malickseneisep2) |
| Ameth FAYE | [@ameth08faye](https://github.com/ameth08faye) |
| Hildegarde Edima BIYENDA| [@HildaEDIMA](https://github.com/HildaEDIMA) |
| Albert ZINABA | [@ZINABA-Albert](https://github.com/ZINABA-Albert) |

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Cloner le repo
git clone https://github.com/Luck-John/Projet-ML-Prediction-of-building-energy.git
cd "Projet-ML-Prediction of building energy"

# Créer virtualenv
python -m venv .venv

# Activer
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Installer dépendances
pip install -r requirements.txt
```

### 2️⃣ Démarrer l'API (FastAPI)

```bash
uvicorn src.api.main:app --reload
# → http://localhost:8000/docs (Swagger UI)
```

### 3️⃣ Démarrer le Dashboard (Streamlit)

```bash
streamlit run src/dashboard/app.py
# → http://localhost:8501
```

### 4️⃣ Lancer les Tests

```bash
pytest tests/ -v
```

### 5️⃣ Réentraîner le Modèle

```bash
python -m src.models.train
```

---

## 📁 Structure du Projet

```
Projet ML-Prediction of building energy/
│
├── 📂 artifacts/                              ✅ MODÈLES & RÉSULTATS
│   ├── model.joblib                           ✅ Model FINAL (24.6 MB)
│   ├── best_params.joblib                     ✅ Hyperparamètres
│   └── compare_report.joblib                  ✅ Rapport comparaison
│
├── 📂 .github/workflows/                      ✅ CI/CD GITHUB ACTIONS
│   └── ci.yml                                 ✅ Pipeline automatique
│
├── 📂 data/                                   ✅ DONNÉES
│   ├── processed/                             ✅ Données traitées
│   │   └── 2016_Building_Energy_Benchmarking.csv
│   └── raw/                                   ✅ Données brutes
│       └── 2016_Building_Energy_Benchmarking.csv
│
├── 📂 notebooks/                              ✅ DOCUMENTATION
│   └── energy_01_analyse (11).ipynb           ✅ RÉFÉRENCE MODÈLE
│
├── 📂 src/                                    ✅ CODE SOURCE
│   ├── api/
│   │   └── main.py                            ✅ API FastAPI
│   ├── dashboard/
│   │   └── app.py                             ✅ Dashboard Streamlit
│   ├── preprocessing/
│   │   └── preprocessor.py                    ✅ Nettoyage données
│   ├── features/
│   │   └── engineer.py                        ✅ Feature engineering
│   └── models/
│       ├── train.py                           ✅ Entraînement stacking
│       ├── evaluate.py                        ✅ Évaluation
│       └── compare_pipelines.py               ✅ Comparaison
│
├── 📂 tests/                                  ✅ TESTS UNITAIRES
│   ├── test_preprocess.py                     ✅ Valide preprocessing
│   ├── test_models.py                         ✅ Valide modèles
│   └── test_integration_metrics.py            ✅ Tests intégration
│
├── 📄 requirements.txt                        ✅ Dépendances Python
├── 📄 pytest.ini                              ✅ Config pytest
├── 📄 .gitignore                              ✅ Fichiers à ignorer
├── 📄 README.md                               ✅ Ce fichier (guide)
├── 📄 ESSENTIAL_FILES.md                      ✅ Guide fichiers clés
└── 📄 CLEANUP_AUDIT.md                        ✅ Audit du projet
```

---

## 🚀 Installation & Setup

### 1. Cloner le projet
```bash
git clone https://github.com/votre-user/Projet-ML-Prediction-of-building-energy.git
cd Projet-ML-Prediction-of-building-energy
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Ajouter le projet au PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
# ou
set PYTHONPATH=%cd%                       # Windows
```

---

## 📊 Données & Prétraitement

### Filtrage Appliqué
- **Bâtiments non-résidentiels uniquement** : Exclusion des multifamilial (1-4, 5-9, 10+)
- **Suppression des valeurs aberrantes** :
  - Consommation > 2×10⁸ kBtu (super-consommateurs)
  - Surface > 3×10⁶ sqft
- **Gestion des valeurs manquantes** :
  - `ENERGYSTARScore` : Imputation par médiane par type de bâtiment
  - Autres : Suppression des lignes

### Transformations
- **Log-transformation** : `SiteEnergyUse_log = log(SiteEnergyUse(kBtu))`
- **Target Encoding** : Variables catégorielles (smoothing=10)
- **Feature Engineering** :
  - Distance au centre-ville
  - Clustering spatial (10 clusters)
  - Indicateur centre-ville (< 2km)
  - Clustering par surface

---

## 🔬 Modèles & Scénarios

### Scénarios Testés
1. **Avec ENERGYSTARScore** : Utilisation complète du score
2. **Sans ENERGYSTARScore** : Exclusion du score (robustesse)

### Familles de Modèles
| Famille | Modèles |
|--------|---------|
| **Linéaire** | LinearRegression, Ridge, Lasso, ElasticNet, LinearSVR |
| **Arbre** | DecisionTree, KNN |
| **Ensemble** | RandomForest, ExtraTrees, XGBoost, LightGBM, HistGradientBoosting |

### Hyperparamètres Optimisés
- **GridSearchCV** (CV=5 pour linéaire, CV=3 pour ensemble)
- **Métrique** : RMSE négative
- **Parallélisation** : n_jobs=-1

---

## 📈 Métriques d'Évaluation

### Métriques Log-Scale
- **RMSE_Log** : Erreur RMS sur l'espace log
- **R²_Log** : Coefficient de détermination log

### Métriques Real-Scale (Prioritaires)
- **MAPE_Real** : Erreur Moyenne en Pourcentage Absolu (critère #1)
- **R²_Real** : Coefficient de détermination réel (critère #2)
- **RMSE_Real** : Erreur RMS réelle (kBtu)
- **MAE_Real** : Erreur Absolue Moyenne (kBtu)

---

## 🔧 Utilisation

### 1. Entraînement Simple
```bash
python -m src.models.train
```

### 2. Évaluation
```bash
python -m src.models.run_evaluation
```

### 3. Comparaison Multi-Modèles
```bash
python -m src.models.compare_pipelines
```

### 4. Tests Unitaires
```bash
pytest tests/ -v
```

### 5. Inférence
```python
from src.models.inference import predict

prediction = predict(X_new)
print(f"Prédiction : {prediction} kBtu")
```

---

## 🔍 MLflow Tracking

Tous les modèles sont automatiquement loggés dans **MLflow** :
```bash
mlflow ui  # Lancer le dashboard (http://127.0.0.1:5000)
```

**Éléments trackés:**
- Hyperparamètres
- Métriques (MAPE, R², RMSE, MAE)
- Modèle serialisé
- Artefacts (encoder, scaler)

---

## 🤖 CI/CD Pipeline

### GitHub Actions (`.github/workflows/ci.yml`)

Exécution automatique sur chaque `git push` :

1. ✅ **Setup Python** : 3.10
2. ✅ **Install Dependencies** : `pip install -r requirements.txt`
3. ✅ **Lint** : Vérification syntaxe avec flake8
4. ✅ **Run Tests** : `pytest tests/`
5. ✅ **Upload Artifacts** : Résultats tests

**Branches:** main, master
**Status:** Visible sur GitHub (badges, actions tab)

---

## 🧪 Structure des Tests

### `test_preprocess.py`
- ✅ Vérification suppression NaNs
- ✅ Présence cible
- ✅ Chargement depuis fichier

### `test_models.py`
- ✅ Existence artefacts modèles
- ✅ Structure correcte (dict avec clé 'model')

### `test_integration_metrics.py`
- ✅ Comparaison notebook vs refactor
- ✅ Tolérance 5% sur MAE

**Exécution locale:**
```bash
pytest tests/ -v --tb=short
```

---

## 📝 Configuration (`configs/params.yaml`)

```yaml
# Ajouter si nécessaire
data:
  raw_path: data/raw/2016_Building_Energy_Benchmarking.csv
  processed_path: data/processed/2016_Building_Energy_Benchmarking.csv

preprocessing:
  test_size: 0.2
  random_state: 42
  target_col: SiteEnergyUse_log

models:
  random_state: 42
  cv_folds: 5
```

---

## 🐛 Dépannage

### Erreur : `ModuleNotFoundError: No module named 'src'`
**Solution :**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Erreur : `ImportError` lors des tests
**Solution :**
```bash
# Assurez-vous que conftest.py ajoute le root au path
pytest tests/ -v
```

### MLflow ne sauvegarde pas
**Solution :**
```bash
# Vérifier l'expérience MLflow
mlflow experiments list
mlflow runs list --experiment-id 0
```

---

## 📚 Ressources

- **Data Source** : [Kaggle - Seattle Building Energy](https://www.kaggle.com/datasets)
- **Scikit-learn** : [Documentation](https://scikit-learn.org/)
- **MLflow** : [Documentation](https://mlflow.org/)
- **XGBoost** : [Documentation](https://xgboost.readthedocs.io/)
- **LightGBM** : [Documentation](https://lightgbm.readthedocs.io/)

---

## 👥 Contributeurs

- **MLOps Engineer / Code Quality** : Votre Nom

---

## 📄 Licence

Ce projet est sous licence MIT. Voir `LICENSE` pour détails.

---

## ✅ Checklist MLOps - Bloc 4

- ✅ Architecture Git structurée (src/, tests/, notebooks/, data/)
- ✅ Refactoring Notebook → Scripts Python
- ✅ MLflow tracking intégré
- ✅ Tests unitaires et intégration
- ✅ CI/CD avec GitHub Actions
- ✅ Gestion des imports (`__init__.py`)
- ✅ Pytest configuré
- ✅ Documentation complète (README)

---

**Dernière mise à jour :** Janvier 2026
