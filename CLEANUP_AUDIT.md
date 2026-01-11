# 🧹 Audit et Nettoyage du Projet ML

## 📊 Structure Actuelle du Projet

```
Projet ML-Prediction of building energy/
├── .github/workflows/       ✅ UTILE - CI/CD
│   └── ci.yml               ✅ Nécessaire pour GitHub Actions
├── artifacts/               ✅ CRITIQUE - Modèles et rapports
│   ├── model.joblib         ✅ FINAL MODEL (24.6 MB)
│   ├── best_params.joblib   ✅ Hyperparams
│   └── compare_report.joblib ✅ Benchmark report
├── configs/                 ⚠️ À ANALYSER
│   └── params.yaml          ⚠️ Obsolète? (pas utilisé actuellement)
├── data/                    ✅ UTILE
│   ├── raw/                 ✅ Source originale (~1.2 MB)
│   └── processed/           ✅ CSV traité (~396 KB)
├── notebooks/               ✅ UTILE
│   ├── energy_01_analyse (11).ipynb  ✅ KEEP (modèle FINAL)
│   └── energy_01_analyse (5).ipynb   ❌ À SUPPRIMER (ancien)
├── mlruns/                  ⚠️ À DÉCIDER - MLflow tracking
│   └── (historique expériences)     (~25 MB)
├── src/                     ✅ CRITIQUE - Code principal
│   ├── preprocessing/       ✅ preprocessor.py
│   ├── features/            ✅ engineer.py
│   └── models/              ✅ train.py, evaluate.py, compare_pipelines.py
├── tests/                   ✅ UTILE - Tests unitaires
│   ├── test_preprocess.py   ✅ Nécessaire
│   ├── test_models.py       ✅ Nécessaire
│   └── test_integration_metrics.py ✅ Nécessaire
├── .venv/                   ⚠️ À DÉCIDER - Virtualenv (1+ GB)
├── .gitignore               ✅ Nécessaire
├── pytest.ini               ✅ Config tests
├── requirements.txt         ✅ Dependencies
├── README.md                ✅ Documentation
└── METRICS_ALIGNMENT.md     ✅ Documentation qualité
```

---

## 🗑️ Ce qui DOIT Être Supprimé

### 1. **energy_01_analyse (5).ipynb** ❌
- Ancien notebook, remplacé par (11)
- **Action:** Supprimer
- **Espace libéré:** ~2.1 MB

### 2. **configs/params.yaml** ❓
- Utilisé? Vérifier dans le code...
- **Action:** Supprimer si inutilisé
- **Espace libéré:** <1 KB (négligeable)

---

## ⚠️ Ce qui EST À DÉCIDER - MLflow

### mlruns/ Dossier (25+ MB)

**QU'EST-CE QUE C'EST?**
- Dossier local créé par MLflow pour tracker les expériences
- Contient: métriques, paramètres, artefacts d'entraînement
- Raison: logging automatique lors de `train.py`

**GARDER ou SUPPRIMER?**

| Aspect | Garde mlruns/ | Supprime mlruns/ |
|--------|--------------|-----------------|
| **Size** | +25 MB | -25 MB (plus léger) |
| **Historique** | ✅ Logs de tous les runs | ❌ Perte d'historique |
| **CI/CD** | Peut regénérer à chaque fois | Moins d'info de debug |
| **Production** | N'est pas utilisé | N'est pas utilisé |
| **Recommandation** | Si tu veux tracker les expériences | Si tu veux un repo léger |

**NOTRE RECOMMANDATION:** ✅ **SUPPRIMER `mlruns/`**
- Tu n'en as pas besoin en production
- Le modèle final est dans `artifacts/model.joblib`
- Si tu veux tracker ultérieurement, utilise une DB (SQLite, Postgres)
- Économise 25 MB

---

##  .venv/ (1-2 GB)

**GARDER LOCALEMENT** ✅ (pour développement)
**IGNORER SUR GIT** ✅ (déjà dans `.gitignore`)

Le `.venv/` n'est **pas uploadé sur GitHub** donc c'est OK. La commande pour recréer:
```bash
python -m venv .venv
pip install -r requirements.txt
```

---

## 🎯 Fichiers CRITIQUES Pour le Modèle

### **Actuel - Fichiers Minimum pour Faire Fonctionner le Modèle:**

```
✅ ESSENTIELS (pour charger et utiliser le modèle):
  artifacts/
  ├── model.joblib              (LE MODÈLE FINAL)
  ├── best_params.joblib        (Hyperparamètres)
  
✅ ESSENTIELS (pour réentraîner):
  src/
  ├── models/train.py           (Script entraînement)
  ├── models/evaluate.py        (Évaluation)
  ├── models/compare_pipelines.py (Comparaison)
  ├── preprocessing/preprocessor.py
  ├── features/engineer.py
  
✅ ESSENTIELS (données):
  data/processed/2016_Building_Energy_Benchmarking.csv
  
✅ ESSENTIELS (configuration):
  requirements.txt
  .github/workflows/ci.yml
  
✅ ESSENTIELS (tests/qualité):
  tests/test_*.py
  pytest.ini

❓ OPTIONNEL:
  notebooks/energy_01_analyse (11).ipynb (référence, pas production)
  README.md (documentation)
  METRICS_ALIGNMENT.md (documentation)
```

### **Pour Utiliser le Modèle en Production:**

```python
import joblib

# Charger le modèle
model_dict = joblib.load("artifacts/model.joblib")
model = model_dict['model']
encoder = model_dict['encoder']

# Prédire
X_new = ...  # tes données
pred = model.predict(X_new)
```

**Fichiers ABSOLUMENT NÉCESSAIRES:**
1. `artifacts/model.joblib` ✅
2. `artifacts/best_params.joblib` ✅ (optionnel, juste info)
3. `data/processed/2016_Building_Energy_Benchmarking.csv` ✅ (si réentraînement)

---

## 🧹 Plan de Nettoyage Recommandé

### **PHASE 1: Suppression Agressive (Économise ~30 MB)**

```powershell
# Supprimer ancien notebook
rm notebooks/energy_01_analyse\ \(5\).ipynb

# Supprimer MLflow runs (optionnel mais recommandé)
rm -r mlruns/

# Supprimer configs inutilisé
rm configs/params.yaml

# Vider __pycache__ (regénéré automatiquement)
# Cela se fait avec: git clean -fd __pycache__
```

### **PHASE 2: Vérifier .gitignore**

Ensure `.gitignore` contains:
```
__pycache__/
*.pyc
*.pyo
.venv/
.env
*.egg-info/
dist/
build/
mlruns/  ← Add this if not present
```

### **PHASE 3: Commit & Push**

```bash
git add -A
git commit -m "Clean up: remove old notebook, mlflow logs, and unused configs"
git push origin master
```

---

## 📈 Résultat Final (Après Nettoyage)

```
Taille avant: ~1.5+ GB (avec .venv)
Taille repo après nettoyage: ~50-100 MB (sans .venv, sans mlruns)

Fichiers critiques maintenus:
✅ Model: artifacts/model.joblib (24.6 MB) - NE PAS TOUCHER
✅ Code: src/ - NE PAS TOUCHER
✅ Tests: tests/ - NE PAS TOUCHER
✅ Data: data/processed/ - NE PAS TOUCHER
✅ Config: .github/, pytest.ini, requirements.txt - NE PAS TOUCHER
```

---

## ✅ Checklist Finale

- [ ] Garder `notebooks/energy_01_analyse (11).ipynb` (référence)
- [ ] Supprimer `notebooks/energy_01_analyse (5).ipynb` (ancien)
- [ ] Supprimer `mlruns/` (MLflow local - pas production)
- [ ] Supprimer `configs/params.yaml` (inutilisé)
- [ ] Garder `artifacts/` (CRITIQUE)
- [ ] Garder `src/` (CRITIQUE)
- [ ] Garder `tests/` (IMPORTANT)
- [ ] Garder `.venv/` localement (pas sur Git)
- [ ] Update `.gitignore` si besoin
- [ ] Commit & Push

---

## 🎯 TL;DR - Action Immédiate

```powershell
# SUPPRIMER CES FICHIERS/DOSSIERS:
cd "C:\Users\...\Projet ML-Prediction of building energy"

# 1. Ancien notebook
rm "notebooks\energy_01_analyse (5).ipynb"

# 2. MLflow logs (optionnel mais recommandé)
rm -r mlruns

# 3. Config inutilisé
rm "configs\params.yaml"

# 4. Commit
git add -A
git commit -m "Cleanup: remove unused files and mlflow logs"
git push origin master
```

**Économies:** ~30 MB de repo, repo plus propre et maintenable ✅
