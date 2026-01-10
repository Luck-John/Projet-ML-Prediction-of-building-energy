# 🏢 Building Energy Prediction - Interface Générale

Ce projet offre **3 façons flexibles** d'utiliser le modèle prédictif sans être limité à une seule interface.

## 🎯 3 Options Disponibles

### **Option 1: Service Python Réutilisable** 
*Pour intégration dans vos scripts Python*

```python
from src.models.prediction_service import PredictionService

# Initialiser le service
service = PredictionService(use_energy_star=True)

# Prédiction simple
record = {
    'PrimaryPropertyType': 'Office',
    'BuildingType': 'Commercial',
    'PropertyGFATotal': 100000.0,
    'YearBuilt': 2005,
    'Latitude': 47.6,
    'Longitude': -122.3,
    'Neighborhood': 'Downtown Seattle',
    'LargestPropertyUseType': 'Office',
    'ListOfAllPropertyUseTypes': 'Office',
    'ENERGYSTARScore': 75.0
}

result = service.predict_single(record)
print(f"Prédiction: {result['prediction_kbtu']:.0f} kBtu")
```

**Avantages:**
- ✅ Zéro dépendances web
- ✅ Intégration facile dans des scripts existants
- ✅ Peut être utilisé en batch (`predict_batch()`)
- ✅ Pour data science, ETL, jobs automatisés

**Fichier:** `src/models/prediction_service.py`

---

### **Option 2: Dashboard Streamlit** 
*Pour interface web interactive sans backend complexe*

```bash
pip install streamlit
streamlit run dashboard.py
```

**Fonctionnalités:**
- 🎨 Interface web moderne et responsive
- 📊 Prédictions simples (formulaire interactif)
- 📂 Prédictions par lot (upload CSV)
- 📚 Documentation intégrée
- 📥 Export des résultats en CSV

**Avantages:**
- ✅ Zéro configuration (pure Python)
- ✅ Parfait pour les démonstrations
- ✅ Déploiement facile (Streamlit Cloud, Docker)
- ✅ Pour analystes, stakeholders non-techniques

**Fichier:** `dashboard.py`

**Lancer localement:**
```bash
cd c:\Users\ASUS\OneDrive\Desktop\ISE\ISE2\Semestre\ 1\Machine\ Learning\ 1\Projet\ ML\Projet\ ML-Prediction\ of\ building\ energy
streamlit run dashboard.py
```

---

### **Option 3: FastAPI (API REST)** 
*Pour intégration dans des applications/services externes*

```bash
pip install fastapi uvicorn
python -m uvicorn src.api.main:app --reload
```

**Endpoints:**
- `GET /` - Info générale
- `GET /health` - Vérifier que l'API fonctionne
- `GET /model-info` - Info sur un modèle
- `POST /predict` - Prédiction simple (JSON)
- `POST /predict-batch` - Prédictions par lot
- `GET /required-columns` - Schéma des données requises
- `GET /docs` - Documentation interactive (Swagger UI)

**Exemple avec curl:**
```bash
curl -X POST "http://localhost:8000/predict?use_energy_star=true" \
  -H "Content-Type: application/json" \
  -d '{
    "PrimaryPropertyType": "Office",
    "BuildingType": "Commercial",
    "PropertyGFATotal": 100000,
    "YearBuilt": 2005,
    "Latitude": 47.6,
    "Longitude": -122.3,
    "Neighborhood": "Downtown",
    "LargestPropertyUseType": "Office",
    "ListOfAllPropertyUseTypes": "Office",
    "ENERGYSTARScore": 75
  }'
```

**Avantages:**
- ✅ Intégration dans des applications web/mobile
- ✅ Requêtes HTTP standard (langage-agnostique)
- ✅ Scaling et déploiement professionnel
- ✅ Pour applications en production

**Fichier:** `src/api/main.py`

---

## 🔄 Comparaison

| Critère | Service Python | Dashboard Streamlit | API FastAPI |
|---------|---|---|---|
| **Complexité** | Très simple | Simple | Moyenne |
| **Interface** | Code | Web UI | HTTP REST |
| **Déploiement** | Local/ETL | Cloud facile | Production prête |
| **Utilisateurs** | Devs | Analystes | Intégrateurs |
| **Dépendances externes** | Aucune | Streamlit | FastAPI + Uvicorn |
| **Scalabilité** | Moderate | Limitée | Excellente |

---

## 📦 Installation des Dépendances

**Toutes les options:**
```bash
pip install pandas numpy scikit-learn category_encoders joblib
```

**Pour Dashboard:**
```bash
pip install streamlit
```

**Pour API:**
```bash
pip install fastapi uvicorn
```

---

## 🎓 Cas d'Usage

### Service Python
```python
# Intégration dans un pipeline ETL
from src.models.prediction_service import PredictionService

service = PredictionService(use_energy_star=False)

# Traiter un lot de bâtiments
buildings = load_from_database()  # Vos données
predictions = service.predict_batch(buildings)
save_to_database(predictions)
```

### Dashboard Streamlit
```
Manager veut voir prédictions interactives
→ streamlit run dashboard.py
→ Ouvre http://localhost:8501
→ Formulaire + résultats instantanés
```

### API FastAPI
```
Application web (React, Vue, Angular) veut prédictions
→ Appelle POST /predict avec données
← Reçoit JSON prédiction
→ Affiche résultat à l'utilisateur
```

---

## 🚀 Déploiement Recommandé

### Développement Local
**Choisir 1 de ces 3:**
- `python src/models/prediction_service.py` (test du service)
- `streamlit run dashboard.py` (test du dashboard)
- `uvicorn src.api.main:app --reload` (test API)

### Production
**Solution 1:** Dashboard sur Streamlit Cloud
```bash
git push  # Push vers GitHub
# → Configuration sur https://streamlit.io/cloud
# → Auto-déploiement gratuit
```

**Solution 2:** API sur Heroku / Railway / Render
```dockerfile
# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Solution 3:** Service intégré dans Python
```python
# Votre application existante
from src.models.prediction_service import PredictionService
service = PredictionService()
```

---

## 📋 Fichiers de Référence

```
├── src/models/prediction_service.py  ← Logique générale
├── src/api/main.py                   ← API FastAPI
├── dashboard.py                       ← Streamlit UI
├── artifacts/
│   ├── best_model_with_score.joblib
│   └── best_model_no_score.joblib
└── src/preprocessing/
    └── preprocessor.py               ← Transformation données
```

---

## ❓ FAQ

**Q: Je dois faire API ou Dashboard?**
A: Essayez les 2 ! Utilisez `PredictionService` dans les deux cas. C'est flexible.

**Q: Puis-je combiner API + Dashboard?**
A: Oui ! Le Dashboard peut appeler l'API via HTTP, ou tous deux peuvent utiliser le Service.

**Q: Quelle option pour production?**
A: API FastAPI + déploiement sur un serveur. Dashboard Streamlit est plus pour démos.

**Q: Comment ajouter une nouvelle interface?**
A: Importez `PredictionService`, ça fonctionne avec n'importe quel framework (Flask, Django, etc.)

---

## 📞 Support

Voir documentation complète dans les fichiers:
- `src/models/prediction_service.py` - Docstrings détaillées
- `dashboard.py` - Commentaires dans le code
- `src/api/main.py` - `/docs` endpoint (Swagger UI)

---

**Version:** 1.0 | **Modèle:** ExtraTreesRegressor | **Précision:** MAPE ≈ 0.40-0.50
