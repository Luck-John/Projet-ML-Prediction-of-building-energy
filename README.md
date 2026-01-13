# Building Energy Prediction - Seattle

Prédire la consommation énergétique des bâtiments non-résidentiels de Seattle avec Machine Learning

## Table des matières

1. [Objectif](#objectif)
2. [Équipe](#équipe)
3. [Structure](#structure)
4. [Installation](#installation)
5. [Méthodologie](#méthodologie)
6. [Données](#données)
7. [Production - Encoder et Artifacts](#production---encoder-et-artifacts)
8. [Modèle](#modèle)
9. [API & Dashboard](#api--dashboard)
10. [Utilisation](#utilisation)
11. [Tests & CI/CD](#tests--cicd)

---

## Objectif

- Prédire la consommation énergétique (kBtu) des bâtiments non-résidentiels
- Evaluer l’intérêt de l’ENERGY STAR Score pour la prédiction de consommation d’énergie
- Dataset : 2016 Building Energy Benchmarking (Seattle) - 1,553 bâtiments
- Modèle déployé avec API REST (FastAPI) et Dashboard (Lovable)

---

## Équipe

| Collaborateur | GitHub |
|---|---|
| Malick SENE | [@malickseneisep2](https://github.com/malickseneisep2) |
| Ameth FAYE | [@ameth08faye](https://github.com/ameth08faye) |
| Hildegarde Edima BIYENDA | [@HildaEDIMA](https://github.com/HildaEDIMA) |
| Albert ZINABA | [@ZINABA-Albert](https://github.com/ZINABA-Albert) |
| Jean Luc BATABATI | [@Luck-John](https://github.com/Luck-John) |

---

## Structure

```
Projet ML-Prediction of building energy/
│
├── artifacts/                          # Modèles entraînés
│   ├── model.joblib              Model final (24.6 MB)
│   ├── model.pkl                 Backup format
│   ├── best_params.joblib        Hyperparamètres optimisés
│   ├── kmeans_neighborhood.joblib    KMeans (10 clusters)
│   ├── kmeans_neighborhood.pkl
│   ├── kmeans_surface.joblib         KMeans (2 clusters)
│   ├── kmeans_surface.pkl
│   └── data_version.json             Versioning données
│
├── .github/workflows/
│   └── ci.yml                    Pipeline CI/CD GitHub Actions
│
├── data/
│   ├── processed/
│   │   └── seattle_energy_cleaned_final.csv
│   └── raw/
│       └── 2016_Building_Energy_Benchmarking.csv
│
├── notebooks/
│   └── energy_01_EDA.ipynb       Analyse exploratory
│
├── src/
│   ├── config.py                 Configuration centralisée
│   ├── mlflow_utils.py           MLflow tracking
│   ├── preprocessing/
│   │   └── preprocessor.py       Nettoyage données + production mode
│   ├── features/
│   │   └── engineer.py           Feature engineering + production
│   └── models/
│       ├── train.py              Entraînement Stacking
│       ├── evaluate.py           Évaluation
│       └── compare_pipelines.py  Comparaison modèles
│
├── tests/
│   ├── unit/                     Tests unitaires
│   │   ├── test_preprocessing.py     Preprocessing functions
│   │   ├── test_features.py          Feature engineering
│   │   └── test_models.py            Model artifacts
│   ├── integration/              Tests d'intégration
│   │   ├── test_pipeline.py          Pipeline complet
│   │   └── test_end_to_end.py        End-to-end
│   ├── conftest.py               Configuration pytest + fixtures
│   └── __init__.py
│
├── mlruns/                        MLflow experiments
├── api/                           API FastAPI
├── requirements.txt               Dépendances Python
├── pytest.ini                     Configuration pytest
├── .gitignore
├── .mlflowignore
└── README.md                      Ce fichier
```

**Organisation des tests:**
- **Unit tests** : Testent les composants individuels (preprocessing, features, models)
- **Integration tests** : Testent le pipeline complet end-to-end
- Tous les tests passent (20+ tests)
- Marqueurs pytest : `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`

**Artifacts gérés:**
- Modèles sauvegardés en joblib (format principal) + pickle (backup)
- KMeans models persisted pour prédictions consistantes
- Versioning données avec SHA256 hash

---

## Installation

```bash
# Cloner
git clone https://github.com/Luck-John/Projet-ML-Prediction-of-building-energy.git
cd "Projet-ML-Prediction of building energy"

# Virtualenv
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac

# Dépendances
pip install -r requirements.txt
```

---

## Méthodologie

1. **Nettoyage** : Filtrage non-résidentiels, suppression aberrantes, imputation NaN
2. **Feature Engineering** : Distance Haversine, clustering spatial, log-transformation (58 features)
3. **Modèles Testés** : Linéaires, arbres, ensemble (RandomForest, XGBoost, LightGBM)
4. **Optimisation** : GridSearchCV (CV=5) sur hyperparamètres
5. **Architecture Finale** : Stacking Regressor (4 base learners + LinearSVR)
6. **Validation** : 5 tests automatisés passants (100%)

---

## Données

- Bâtiments non-résidentiels uniquement
- Suppression outliers : consommation > 2×10⁸ kBtu, surface > 3×10⁶ sqft
- Imputation ENERGYSTARScore par médiane par type
- Split : 80% train (1,242), 20% test (311)

Transformations :
- Log(SiteEnergyUse)
- Target Encoding (catégories)
- Features géographiques

---

## Production - Encoder et Artifacts

### ⚠️ IMPORTANT: Encoder pour Production

L'encodage catégorique est **CRITIQUE** pour la production. Voici ce qui est sauvegardé:

#### 1. **TargetEncoder** (artifacts/model.joblib)
```
- Type: category_encoders.TargetEncoder
- Configuration: handle_unknown='value'
- Colonnes: ['BuildingType', 'PrimaryPropertyType', 'ZipCode', 'CouncilDistrictCode',
              'Neighborhood', 'ListOfAllPropertyUseTypes', 'LargestPropertyUseType', 
              'Surface_Cluster']
```

**Pourquoi handle_unknown='value' ?**
L'encodeur utilise la **MOYENNE de la target** pour chaque catégorie. En production, si une catégorie est inconnue, l'encodeur remplace automatiquement par une valeur de remplacement. Cela évite les crashes.

#### 2. **Modèles KMeans** (pré-entraînés)
```
- artifacts/kmeans_neighborhood.joblib    → 10 clusters (latitude/longitude)
- artifacts/kmeans_surface.joblib         → 2 clusters (log surface)
```

**IMPORTANT :** Ne pas réentraîner ! Charger et utiliser avec `.predict()`, pas `.fit_predict()`

### Comment charger en production:

```python
from src.preprocessing.production_artifacts import load_all_artifacts

artifacts = load_all_artifacts()
encoder = artifacts['encoder']
kmeans_neighborhood = artifacts['kmeans_neighborhood']
kmeans_surface = artifacts['kmeans_surface']
model = artifacts['model']
```

### Checklist Production:

- ✅ Normaliser catégories (lowercase) **AVANT** encodage
- ✅ Créer toutes les features (24 au total)
- ✅ Appliquer encoder sauvegardé (handle_unknown='value')
- ✅ Charger KMeans pré-entraînés (predict, pas fit_predict)
- ✅ Ne pas modifier les modèles en production

📚 **Documentation détaillée:** Voir [ENCODER_PRODUCTION_GUIDE.md](ENCODER_PRODUCTION_GUIDE.md)

---

## Modèle

**Architecture :** StackingRegressor

**Base Learners (4) :**
- ExtraTreesRegressor (n_est=500, max_depth=10)
- XGBRegressor (n_est=300, learning_rate=0.05, depth=3)
- LGBMRegressor (n_est=100, learning_rate=0.05, num_leaves=50)
- HistGradientBoostingRegressor (learning_rate=0.05, max_iter=200)

**Meta-Learner :** LinearSVR (C=10, dual='auto')

**Performance :** MAPE 0.42, R² 0.527, pas overfitting

---

## API & Dashboard

### API REST (FastAPI)

```bash
uvicorn src.api.main:app --reload
```

Endpoints : `/health`, `/predict`, `/metrics`
Swagger UI : http://localhost:8000/docs

### Dashboard (Lovable)

Interface web interactif avec prédictions, visualisations et comparaisons

```bash
streamlit run src/dashboard/app.py
```

Accès : http://localhost:8501

---

## Utilisation

```bash
# Entraîner
python -m src.models.train

# Évaluer
python -m src.models.evaluate

# Comparaison modèles
python -m src.models.compare_pipelines

# Tests
pytest tests/ -v

# MLflow tracking
mlflow ui  # http://127.0.0.1:5000
```

---

## Tests & CI/CD

### Tests Locaux

**Structure organisée:**
- **Unit tests** : Composants individuels (preprocessing, features, models)
- **Integration tests** : Pipeline complet end-to-end

**Exécution:**
```bash
# Tous les tests
pytest tests/ -v

# Uniquement unit tests
pytest tests/unit/ -v -m unit

# Uniquement integration tests
pytest tests/integration/ -v -m integration

# Test spécifique
pytest tests/unit/test_preprocessing.py -v

# Avec coverage
pytest tests/ --cov=src --cov-report=html
```

**Status:** 20+ tests, 100% passants ✅

### CI/CD Pipeline

GitHub Actions automatise:
1. Setup Python 3.10
2. Installation dépendances
3. Lint avec flake8
4. Training model
5. Exécution tests
6. Upload artifacts

Trigger: Push sur master/main, Pull requests

Logs: `.github/workflows/ci.yml`

---

## Troubleshooting

| Problème | Solution |
|---|---|
| Import errors | `python -m pytest` (pas juste `pytest`) |
| Slow tests | `pytest -m "not slow"` |
| Coverage gaps | `pytest --cov=src --cov-report=html` |
| Test fails on CI | Vérifier Python version et PYTHONPATH |

---

## Configuration

**Centre Seattle:** 47.6062°N, -122.3321°W
**Random State:** 42 (reproducibilité)
**MLflow URI:** `file:./mlruns`
**Expérience:** `building-energy-prediction`

Voir `src/config.py` pour toutes les constantes.
