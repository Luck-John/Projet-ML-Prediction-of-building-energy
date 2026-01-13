# Building Energy Prediction - Seattle

Prédire la consommation énergétique des bâtiments non-résidentiels de Seattle avec Machine Learning

## Table des matières

1. [Objectif](#1-objectif)
2. [Équipe](#2-équipe)
3. [Structure](#3-structure)
4. [Installation](#4-installation)
5. [Méthodologie](#5-méthodologie)
6. [Données](#6-données)
7. [Modèle](#7-modèle)
8. [API et Dashboard](#8-api-et-dashboard)
9. [Utilisation](#9-utilisation)
10. [Tests et CI/CD](#10-tests-et-cicd)
11. [Ressources](#11-ressources)

---

## 1. Objectif

* Prédire la consommation énergétique (kBtu) des bâtiments non-résidentiels
* Evaluer l'intérêt de l'ENERGY STAR Score pour la prédiction de consommation d'énergie
* Dataset : 2016 Building Energy Benchmarking (Seattle) - 1,553 bâtiments
* Modèle déployé avec API REST (FastAPI) et Dashboard (Lovable)

---

## 2. Équipe

| Collaborateur | GitHub |
|---|---|
| Malick SENE | [@malickseneisep2](https://github.com/malickseneisep2) |
| Ameth FAYE | [@ameth08faye](https://github.com/ameth08faye) |
| Hildegarde Edima BIYENDA | [@HildaEDIMA](https://github.com/HildaEDIMA) |
| Albert ZINABA | [@ZINABA-Albert](https://github.com/ZINABA-Albert) |
| Jean Luc BATABATI | [@Luck-John](https://github.com/Luck-John) |

---

## 3. Structure

```
Projet ML-Prediction of building energy/
│
├── artifacts/
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
│   ├── energy_01_EDA.ipynb       Analyse exploratory
│   ├── energy_02_modeling.ipynb  Modèle Stacking
│   └── comparison_notebook_vs_mlops.ipynb  Comparaison résultats
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
├── api/
│   ├── main.py                   API FastAPI
│   ├── requirements.txt           Dépendances API
│   ├── artifacts/
│   │   └── model.joblib
│   └── src/
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
├── requirements.txt               Dépendances Python
├── pytest.ini                     Configuration pytest
├── .gitignore
├── .mlflowignore
└── README.md                      Ce fichier
```

---

## 4. Installation

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

## 5. Méthodologie

1. Nettoyage : Filtrage non-résidentiels, suppression aberrantes, imputation NaN
2. Feature Engineering : Distance Haversine, clustering spatial, log-transformation (58 features)
3. Modèles Testés : Linéaires, arbres, ensemble (RandomForest, XGBoost, LightGBM)
4. Optimisation : GridSearchCV (CV=5) sur hyperparamètres
5. Architecture Finale : Stacking Regressor (4 base learners + LinearSVR)
6. Validation : 20+ tests automatisés passants (100%)

---

## 6. Données

* Bâtiments non-résidentiels uniquement
* Suppression outliers : consommation > 2×10⁸ kBtu, surface > 3×10⁶ sqft
* Imputation ENERGYSTARScore par médiane par type
* Split : 80% train (1,242), 20% test (311)

Transformations :
* Log(SiteEnergyUse)
* Target Encoding (catégories) avec handle_unknown='value' pour production
* Features géographiques (distance Haversine, clustering)

---

## 7. Modèle

Architecture : StackingRegressor

Base Learners (4) :
* ExtraTreesRegressor (max_depth=10, n_estimators=100)
* XGBRegressor (learning_rate=0.05, max_depth=3, n_estimators=300)
* LGBMRegressor (learning_rate=0.05, n_estimators=100, num_leaves=50)
* HistGradientBoostingRegressor (learning_rate=0.05, max_iter=200)

Meta-Learner : LinearSVR (C=10, dual='auto')

Performance :

| Métrique | Train | Test |
|----------|-------|------|
| R2 | 0.8697 (87%) | 0.5141 (51%) |
| MAPE | 24.24% | 40.53% |
| MAE | 1.68M kBtu | 2.47M kBtu |
| RMSE | 4.71M kBtu | 7.99M kBtu |

Artifacts Sauvegardés:
* model.joblib - Modèle Stacking entraîné
* encoder - TargetEncoder avec handle_unknown='value' (production-ready)
* kmeans_geo - KMeans avec 10 clusters (géographie)
* kmeans_surf - KMeans avec 2 clusters (surface)
* training_columns - Liste des 22 features pour validation
* best_params - Hyperparamètres optimisés

---

## 8. API et Dashboard

### 8.1 API REST (FastAPI) - PRODUCTION

Lien de l'API: https://api-production-aaf4.up.railway.app/docs

Documentation interactive Swagger UI avec tous les endpoints:
* /health - Vérifier l'état du serveur
* /predict - Prédire la consommation énergétique
* /metrics - Obtenir les métriques du modèle

Utilisation locale:
```bash
cd api/
pip install -r requirements.txt
uvicorn main:app --reload
# Accès: http://localhost:8000/docs
```

### 8.2 Dashboard Interactif (Lovable)

Lien du Dashboard: https://senenergy.lovable.app/

Interface web complète avec 5 onglets principaux:

A propos
- Vue d'ensemble du projet
- Guide de navigation
- Fonctionnalités (filtres multi-variables, analyses intelligentes, export rapports HTML)

Vue d'Ensemble
- KPI Synthétique (nombre bâtiments, consommation moyenne, score ENERGY STAR moyen, surface moyenne)
- Analyses automatiques et recommandations pour les modèles

Analyse Univariée
- Statistiques descriptives (moyenne, médiane, écart-type, quartiles)
- Histogrammes de distribution
- Détection d'outliers et transformations nécessaires

Analyse Bivariée
- Nuages de points (scatter plots) et box plots par catégorie
- Identification des prédicteurs potentiels
- Validation des hypothèses de linéarité

Corrélations
- Matrice de corrélation entre variables numériques
- Sélection de features
- Détection de multicolinéarité

---

## 9. Utilisation

```bash
# Entraîner le modèle
python -m src.models.train

# Évaluer le modèle
python -m src.models.evaluate

# Comparaison de modèles
python -m src.models.compare_pipelines

# Tests unitaires et intégration
pytest tests/ -v

# Tests spécifiques
pytest tests/unit/ -v -m unit
pytest tests/integration/ -v -m integration

# MLflow tracking
mlflow ui  # http://127.0.0.1:5000

# API locale
cd api/
uvicorn main:app --reload
```

---

## 10. Tests et CI/CD

### 10.1 Tests Locaux

Structure organisée:
* Unit tests : Composants individuels (preprocessing, features, models)
* Integration tests : Pipeline complet end-to-end

Exécution:
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

Status: 20+ tests, 100% passants

### 10.2 CI/CD Pipeline (GitHub Actions)

GitHub Actions automatise:
1. Setup Python 3.10
2. Installation dépendances
3. Lint avec flake8
4. Training model
5. Exécution tests
6. Upload artifacts

Trigger: Push sur master/main, Pull requests

Logs: .github/workflows/ci.yml

---

## 11. Ressources

### Documentation et Présentation

* Présentation du Projet: https://docs.google.com/presentation/d/1UvH_sBAgbNlDLXT389NaBemeXAYwR2vWFh43RhTWuHM/edit?usp=sharing
* API Documentation: https://api-production-aaf4.up.railway.app/docs
* Dashboard: https://senenergy.lovable.app/

### Notebooks d'Analyse

* energy_01_EDA.ipynb - Analyse exploratoire des données
* energy_02_modeling.ipynb - Développement et tuning du modèle Stacking
* comparison_notebook_vs_mlops.ipynb - Validation de la cohérence Notebook vs MLOps

### Repository GitHub

https://github.com/Luck-John/Projet-ML-Prediction-of-building-energy

---

## Configuration

Centre Seattle: 47.6062°N, -122.3321°W
Random State: 42 (reproducibilité)
MLflow URI: file:./mlruns
Expérience: building-energy-prediction

Voir src/config.py pour toutes les constantes.

---

## Troubleshooting

| Problème | Solution |
|---|---|
| Import errors | python -m pytest (pas juste pytest) |
| Slow tests | pytest -m "not slow" |
| Coverage gaps | pytest --cov=src --cov-report=html |
| Test fails on CI | Vérifier Python version et PYTHONPATH |
| API déploiement | Vérifier Railway credentials et variables d'env |

