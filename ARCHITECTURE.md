# Architecture du Projet

## 📐 Vue d'ensemble

```
project-root/
│
├── 📂 .github/
│   └── workflows/
│       └── ci.yml                    # Pipeline GitHub Actions
│
├── 📂 artifacts/                     # Modèles et données sérialisés
│   ├── best_model_*.joblib           # Meilleur modèle entraîné
│   ├── multi_model_ranking_*.joblib  # Ranking multi-modèles
│   ├── X_train.joblib, X_test.joblib # Features train/test
│   └── y_train.joblib, y_test.joblib # Cibles train/test
│
├── 📂 configs/                       # Configurations
│   └── params.yaml                   # Hyperparamètres
│
├── 📂 data/                          # Données
│   ├── raw/                          # Données brutes
│   │   └── 2016_Building_Energy_Benchmarking.csv
│   └── processed/                    # Données nettoyées
│       └── 2016_Building_Energy_Benchmarking.csv
│
├── 📂 notebooks/                     # Prototypage / EDA
│   └── energy_01_analyse (5).ipynb   # Notebook principal
│
├── 📂 src/                           # Code source (refactorisé)
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                   # API REST (FastAPI)
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py               # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                  # Entraînement + MLflow
│   │   ├── evaluate.py               # Évaluation modèles
│   │   ├── inference.py              # Prédictions
│   │   ├── compare_pipelines.py      # Comparaison modèles
│   │   ├── prediction_service.py     # Service prédiction
│   │   ├── run_evaluation.py         # Pipeline évaluation
│   │   └── multi_evaluate.py         # Multi-scénarios
│   └── preprocessing/
│       ├── __init__.py
│       └── preprocessor.py           # Nettoyage données
│
├── 📂 tests/                         # Tests automatisés
│   ├── __init__.py
│   ├── conftest.py                   # Fixtures pytest
│   ├── test_preprocess.py            # Tests nettoyage
│   ├── test_models.py                # Tests modèles
│   └── test_integration_metrics.py   # Tests intégration
│
├── 📄 .gitignore                     # Fichiers ignorés Git
├── 📄 README.md                      # Documentation projet
├── 📄 TESTING.md                     # Guide tests
├── 📄 ARCHITECTURE.md                # Ce fichier
├── 📄 requirements.txt               # Dépendances Python
└── 📄 pytest.ini                     # Config pytest
```

---

## 🏗️ Composants Principaux

### 1. **Preprocessing** (`src/preprocessing/preprocessor.py`)

**Responsabilité :** Nettoyage et préparation des données.

**Fonctions clés :**
- `preprocess_df(df)` : Nettoyage basique
- `preprocess_data(path)` : Chargement + nettoyage

**Étapes :**
1. Filtrage bâtiments non-résidentiels
2. Suppression colonnes non pertinentes
3. Gestion valeurs manquantes (imputation ENERGYSTARScore)
4. Suppression valeurs aberrantes

---

### 2. **Feature Engineering** (`src/features/engineer.py`)

**Responsabilité :** Création de features pour meilleure modélisation.

**Features créées :**
- `SiteEnergyUse_log` : Log-transformation cible
- `PropertyGFATotal_log` : Log-transformation surface
- `Distance_to_Center` : Distance centre Seattle
- `Neighborhood_Cluster` : Clustering spatial
- `Is_Downtown` : Indicateur centre-ville
- `BuildingAge` : Âge du bâtiment
- `Surface_Cluster` : Clustering par taille

**Encodage :**
- Target Encoding pour catégories (smoothing=10)

---

### 3. **Modélisation** (`src/models/`)

**Composants :**

#### `train.py`
```python
def train_model(use_energy_star=True, mlflow_experiment="default"):
    # 1. Chargement données
    # 2. Prétraitement + Feature engineering
    # 3. Split train/test
    # 4. Entraînement modèle
    # 5. MLflow tracking
    # 6. Sauvegarde artefacts
```

#### `evaluate.py`
```python
def evaluate_model(model, X_test, y_test):
    # 1. Prédictions
    # 2. Calcul métriques (RMSE, MAE, MAPE, R²)
    # 3. Retour rapides
```

#### `inference.py`
```python
def predict(X_new):
    # 1. Chargement modèle depuis artifacts/
    # 2. Prédictions
    # 3. Post-traitement (exp() pour log-scale)
```

#### `compare_pipelines.py`
```python
# Entraînement multi-modèles
# Scénarios : Avec/Sans ENERGYSTARScore
# Comparaison MAPE/R²
```

---

### 4. **Tests** (`tests/`)

#### `conftest.py`
Ajoute le chemin racine au sys.path pour imports relatifs.

#### `test_preprocess.py`
```python
def test_preprocess_df_no_nans():
    # Vérifie suppression NaN ✅

def test_preprocess_from_path():
    # Vérifie chargement CSV ✅
```

#### `test_models.py`
```python
def test_model_artifact_exists():
    # Vérifie existence modèle

def test_model_contains_model_key():
    # Vérifie structure dict
```

#### `test_integration_metrics.py`
```python
def test_refactored_metrics_close_to_notebook():
    # Vérifie cohérence notebook vs refactor
    # Tolérance < 5%
```

---

### 5. **API** (`src/api/main.py`)

Service REST pour prédictions en ligne.

**Endpoints :**
- `POST /predict` : Prédiction simple
- `GET /health` : Vérification santé

---

## 🔄 Pipeline MLOps

### 1. **Développement Local**
```
Notebook → EDA + Expérimentation
   ↓
Refactoring en Scripts Python
   ↓
Entraînement + MLflow Tracking
```

### 2. **Testing**
```
Git Push
   ↓
GitHub Actions Triggered
   ↓
pytest - Lint - Unit Tests
   ↓
Rapport d'erreurs / Succès
```

### 3. **Production**
```
Meilleur modèle → Artefacts
   ↓
API Déploiement
   ↓
Monitoring + Predictions
```

---

## 📊 Flux de Données

### Train Flow
```
Raw Data → Preprocess → Feature Engineer → Encode
   ↓
Train/Test Split
   ↓
Model Training (GridSearchCV)
   ↓
MLflow Logging
   ↓
Artifacts Saved
```

### Inference Flow
```
New Data → Preprocess → Feature Engineer → Encode
   ↓
Load Model from Artifacts
   ↓
Predict (log-scale)
   ↓
Post-process (exp)
   ↓
Return kBtu
```

---

## 🔧 Dépendances & Versions

```
pandas>=1.3.0          # Data manipulation
numpy>=1.21.0          # Numerical computing
scikit-learn>=1.0.0    # ML algorithms
mlflow>=2.0.0          # Experiment tracking
joblib>=1.1.0          # Serialization
category_encoders>=2.5 # Target encoding
xgboost>=1.5.0         # Gradient boosting
lightgbm>=3.3.0        # Light GBM
pytest>=7.0.0          # Testing
```

---

## 🎯 Points d'Entrée Principaux

### 1. **Entraînement**
```bash
python -m src.models.train --use-energy-star
```

### 2. **Évaluation**
```bash
python -m src.models.run_evaluation
```

### 3. **Comparaison Modèles**
```bash
python -m src.models.compare_pipelines
```

### 4. **Tests**
```bash
pytest tests/ -v
```

### 5. **API**
```bash
uvicorn src.api.main:app --reload
```

---

## 📈 MLflow Tracking

Tous les modèles loggent :
- **Paramètres** : Hyperparamètres modèle
- **Métriques** : MAPE, R², RMSE, MAE
- **Modèle** : Sérialisation sklearn
- **Artefacts** : Encoder, Scaler, etc.

**Accès :**
```bash
mlflow ui  # http://127.0.0.1:5000
```

---

## ✅ Critères de Qualité

| Critère | Target | Status |
|---------|--------|--------|
| Test Coverage | > 80% | ⚠️ À améliorer |
| MAPE (Real) | < 15% | ✅ Atteint |
| R² (Real) | > 0.85 | ✅ Atteint |
| CI/CD Pass Rate | 100% | ✅ OK |
| Code Style | PEP8 | ✅ OK |

---

## 🔐 Bonnes Pratiques Appliquées

✅ **Modularité** : Séparation concerns (preprocess, features, models)
✅ **Reproducibilité** : Random states, versioning données
✅ **Testing** : Unit + Integration tests
✅ **Monitoring** : MLflow tracking
✅ **Documentation** : README, TESTING, ARCHITECTURE
✅ **CI/CD** : GitHub Actions pipeline
✅ **Version Control** : .gitignore, commits atomiques

---

## 🚀 Améliorations Futures

- [ ] Ajouter logging robuste
- [ ] Implémenter data validation (Great Expectations)
- [ ] Monitoring modèle en production (Evidently AI)
- [ ] Docker containerization
- [ ] Kubernetes orchestration
- [ ] Feature Store integration
- [ ] A/B Testing framework

---

**Dernière mise à jour :** Janvier 2026
