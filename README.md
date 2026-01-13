# Building Energy Prediction - Seattle

Prédire la consommation énergétique des bâtiments non-résidentiels de Seattle avec Machine Learning

## Table des Matières

1. [Objectif](#objectif)
2. [Équipe](#équipe)
3. [Structure](#structure)
4. [Installation](#installation)
5. [Méthodologie](#méthodologie)
6. [Données](#données)
7. [Modèle](#modèle)
8. [API & Dashboard](#api--dashboard)
9. [Utilisation](#utilisation)
10. [Tests & CI/CD](#tests--cicd)

---

## Objectif

- Prédire la consommation énergétique (kBtu) des bâtiments non-résidentiels
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
├── artifacts/
│   ├── model.joblib              Model final (24.6 MB)
│   ├── best_params.joblib        Hyperparamètres optimisés
│   └── compare_report.joblib     Rapport comparaison
│
├── .github/workflows/
│   └── ci.yml                    Pipeline CI/CD GitHub Actions
│
├── data/
│   ├── processed/
│   │   └── 2016_Building_Energy_Benchmarking.csv
│   └── raw/
│       └── 2016_Building_Energy_Benchmarking.csv
│
├── notebooks/
│   └── energy_01_analyse (11).ipynb     Référence du modèle
│
├── src/
│   ├── api/
│   │   └── main.py                      API FastAPI
│   ├── dashboard/
│   │   └── app.py                       Dashboard Lovable
│   ├── preprocessing/
│   │   └── preprocessor.py              Nettoyage données
│   ├── features/
│   │   └── engineer.py                  Feature engineering
│   └── models/
│       ├── train.py                     Entraînement Stacking
│       ├── evaluate.py                  Évaluation
│       └── compare_pipelines.py         Comparaison modèles
│
├── tests/
│   ├── test_preprocess.py               Tests preprocessing
│   ├── test_models.py                   Tests modèle
│   ├── test_integration_metrics.py      Tests intégration
│   └── conftest.py                      Configuration pytest
│
├── requirements.txt                     Dépendances Python
├── pytest.ini                           Configuration pytest
├── .gitignore                           Fichiers à ignorer
└── README.md                            Ce fichier
```

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

**Tests (5)** : Prétraitement, modèle, intégration - 100% passants

**CI/CD :** GitHub Actions automatise train → test → upload artifacts sur chaque push

---

**Status :** Production-Ready (Bloc 4 Complet)
**Dernière mise à jour :** Janvier 2026

` ash
# Cloner
git clone https://github.com/Luck-John/Projet-ML-Prediction-of-building-energy.git
cd "Projet-ML-Prediction of building energy"

# Virtualenv
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac

# Dépendances
pip install -r requirements.txt
` 

---

## Structure

` 
src/
 preprocessing/     Nettoyage des données
 features/          Feature engineering
 models/            Train, évaluation, comparaison
 api/               API REST (FastAPI)
 dashboard/         Dashboard (Lovable)

tests/                 5 tests unitaires (100% passants)
artifacts/             Modèle et hyperparamètres
notebooks/             Référence du notebook initial
` 

---

## Méthodologie

1. **Nettoyage** : Filtrage non-résidentiels, suppression aberrantes, imputation NaN
2. **Feature Engineering** : Distance Haversine, clustering spatial, log-transformation (58 features)
3. **Modèles Testés** : Linéaires, arbres, ensemble (RandomForest, XGBoost, LightGBM)
4. **Optimisation** : GridSearchCV (CV=5) sur hyperparamètres
5. **Architecture Finale** : Stacking Regressor (4 base learners + LinearSVR)
6. **Validation** : 5 tests automatisés passants

---

## Données

- Bâtiments non-résidentiels uniquement
- Suppression outliers : consommation > 210 kBtu, surface > 310 sqft
- Imputation ENERGYSTARScore par médiane par type
- Split : 80% train (1,242), 20% test (311)

Transformations :
- Log(SiteEnergyUse)
- Target Encoding (catégories)
- Features géographiques

---

## Modèle

**Architecture :** StackingRegressor

**Base Learners (4) :**
- ExtraTreesRegressor (n_est=500)
- XGBRegressor (n_est=300)
- LGBMRegressor (n_est=100)
- HistGradientBoostingRegressor (iter=200)

**Meta-Learner :** LinearSVR (C=10)

**Performance :** MAPE 0.42, R² 0.527, pas overfitting

---

## API & Dashboard

### API REST (FastAPI)

` ash
uvicorn src.api.main:app --reload
` 

Endpoints : ` /health` , ` /predict` , ` /metrics`
Swagger UI : http://localhost:8000/docs

### Dashboard (Lovable)

Interface web interactif avec prédictions, visualisations et comparaisons

` ash
streamlit run src/dashboard/app.py
` 

Accès : http://localhost:8501

---

## Utilisation

` ash
# Entraîner
python -m src.models.train

# Évaluer
python -m src.models.evaluate

# Tests
pytest tests/ -v

# MLflow tracking
mlflow ui  # http://127.0.0.1:5000
` 

---

## Tests & CI/CD

**Tests (5)** : Prétraitement, modèle, intégration - 100% passants

**CI/CD :** GitHub Actions automatise train  test  upload sur chaque push

---

**Status :** Production-Ready (Bloc 4 Complet)
**Mise à jour :** Janvier 2026
