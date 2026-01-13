"""
ENCODER ET ARTIFACTS PRODUITS PAR LE TRAINING

Ce document explique les artifacts sauvegardés et comment les utiliser en production.

==============================================
1. ENCODER (TargetEncoder)
==============================================

📍 Localisation: artifacts/model.joblib (à l'intérieur du dictionnaire)

Description:
- Type: category_encoders.TargetEncoder
- Colonnes encodées: Toutes les colonnes object du dataset
  ['BuildingType', 'PrimaryPropertyType', 'ZipCode', 'CouncilDistrictCode', 
   'Neighborhood', 'LargestPropertyUseType', 'ListOfAllPropertyUseTypes']
- Configuration CRITIQUE: handle_unknown='value'
- Comportement: L'encoder PEUT gérer des catégories inconnues (non vues pendant l'entraînement)

⚠️ POURQUOI C'EST IMPORTANT:
L'encoder utilise la MOYENNE de la TARGET pour chaque catégorie. Si une valeur de catégorie
est inconnue en prédiction, l'encoder replace automatiquement par une valeur de remplacement
grâce à handle_unknown='value'.

Comment charger l'encoder:
```python
import joblib

model_dict = joblib.load('artifacts/model.joblib')
encoder = model_dict['encoder']

# Les colonnes à encoder:
encoder_cols = encoder.cols  # Liste des colonnes

# Transformer les données:
X_transformed = encoder.transform(X_input[encoder_cols])
```

==============================================
2. MODÈLES KMEANS (Clustering)
==============================================

📍 Localisation: 
- artifacts/kmeans_neighborhood.joblib  (10 clusters sur latitude/longitude)
- artifacts/kmeans_surface.joblib       (2 clusters sur surface log)

Description:
- Type: sklearn.cluster.KMeans
- Paramètres: random_state=42, n_init=10
- Utilisation: Feature engineering pour créer Neighborhood_Cluster et Surface_Cluster

⚠️ IMPORTANT - NE PAS RÉENTRAÎNER:
Les modèles KMeans DOIVENT être chargés et utilisés pour la PRÉDICTION.
Ne pas créer de nouveaux KMeans avec fit_predict() car cela changerait les clusters!

Comment charger et utiliser:
```python
import joblib
import numpy as np

# Charger les modèles pré-entraînés
kmeans_neighborhood = joblib.load('artifacts/kmeans_neighborhood.joblib')
kmeans_surface = joblib.load('artifacts/kmeans_surface.joblib')

# Utiliser pour prédiction (predict, pas fit_predict!):
cluster_neighborhood = kmeans_neighborhood.predict(df[['Latitude', 'Longitude']])
cluster_surface = kmeans_surface.predict(df[['PropertyGFATotal_log']])
```

==============================================
3. PIPELINE COMPLET EN PRODUCTION
==============================================

ORDRE CRITIQUE:

1. Normaliser les catégories (LOWERCASE)
   ```python
   for col in ['BuildingType', 'PrimaryPropertyType', ...]:
       df[col] = df[col].str.lower()
   ```

2. Créer les features numériques:
   - BuildingAge = 2016 - YearBuilt
   - PropertyGFATotal_log = log(PropertyGFATotal)
   - Distance_to_Center (haversine distance)
   - Rotated_Lat, Rotated_Lon (30 degrés rotation)

3. APPLIQUER L'ENCODER sur toutes les colonnes catégorielles:
   ```python
   encoder = model_dict['encoder']
   df[encoder.cols] = encoder.transform(df[encoder.cols])
   ```

4. APPLIQUER LES KMEANS (predict, pas fit_predict):
   ```python
   df['Neighborhood_Cluster'] = kmeans_neighborhood.predict(df[['Lat', 'Lon']])
   df['Surface_Cluster'] = kmeans_surface.predict(df[['PropertyGFATotal_log']])
   ```

5. Sélectionner toutes les colonnes finales (24 features)
6. Prédire avec le modèle

⚠️ ERREURS COMMUNES:

❌ MAUVAIS: Réentraîner les KMeans
```python
kmeans = KMeans(n_clusters=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Lat', 'Lon']])  # ❌ WRONG!
```

✅ BON: Charger et utiliser les modèles sauvegardés
```python
kmeans = joblib.load('artifacts/kmeans_neighborhood.joblib')
df['Cluster'] = kmeans.predict(df[['Lat', 'Lon']])  # ✅ CORRECT
```

❌ MAUVAIS: Encoder sans normalisation
```python
df['BuildingType'] = 'DOWNTOWN'
df['BuildingType'] = encoder.transform(df['BuildingType'])  # Peut échouer
```

✅ BON: Encoder après normalisation
```python
df['BuildingType'] = df['BuildingType'].str.lower()  # Normalize first
df['BuildingType'] = encoder.transform(df['BuildingType'])  # Then encode
```

==============================================
4. FICHIERS ARTIFACTS SAUVEGARDÉS
==============================================

artifacts/model.joblib
├── 'model': StackingRegressor (le modèle ML)
├── 'encoder': TargetEncoder (l'encodeur avec handle_unknown='value')
├── 'best_params': dict (hyperparamètres optimisés)
└── 'target_col': 'SiteEnergyUse_log'

artifacts/model.pkl
└── Même contenu que model.joblib (format pickle)

artifacts/kmeans_neighborhood.joblib
└── KMeans avec 10 clusters (latitude/longitude)

artifacts/kmeans_surface.joblib
└── KMeans avec 2 clusters (log surface)

artifacts/best_params.joblib
└── Hyperparamètres optimisés (sauvegardé séparément)

==============================================
5. VARIABLES D'ENVIRONNEMENT (pour l'API)
==============================================

MODEL_PATH = "artifacts/model.joblib"
KMEANS_NEIGHBORHOOD_PATH = "artifacts/kmeans_neighborhood.joblib"
KMEANS_SURFACE_PATH = "artifacts/kmeans_surface.joblib"

==============================================
6. CONFIGURATION DE L'ENCODER EN DÉTAIL
==============================================

```python
encoder = ce.TargetEncoder(
    cols=cat_cols,          # ['BuildingType', 'PrimaryPropertyType', ...]
    smoothing=10,           # Régularisation pour éviter l'overfitting
    handle_unknown='value'  # ⭐ IMPORTANT: Peut gérer catégories inconnues
)
```

La configuration handle_unknown='value' signifie:
- Si une catégorie est inconnue en prédiction, l'encoder retourne une valeur de remplacement
- Cette valeur de remplacement est la MÉDIANE des encodages d'entraînement
- Cela évite les crashes lors de la prédiction sur de nouvelles données

==============================================
7. CHECKLIST AVANT DÉPLOIEMENT
==============================================

✅ Encoder chargé depuis model.joblib
✅ Encoder a handle_unknown='value' configuré
✅ Colonnes catégorielles normalisées (lowercase)
✅ KMeans neighborhood chargé (pas réentraîné)
✅ KMeans surface chargé (pas réentraîné)
✅ Toutes les 24 features créées
✅ Encoder appliqué APRÈS normalisation
✅ KMeans appliqué APRÈS feature engineering
✅ Modèle prêt pour prédiction

==============================================
8. SUPPORT
==============================================

Si l'encoder échoue à transformer une colonne:
- Vérifier que la colonne est en string, pas en int
- Vérifier que la colonne a été normalisée (lowercase)
- Vérifier que encoder.cols contient le nom de la colonne
- Vérifier que le fichier model.joblib n'est pas corrompu

Si les KMeans échouent:
- Vérifier qu'on utilise predict(), pas fit_predict()
- Vérifier que les colonnes d'entrée existent et sont numériques
- Vérifier que le nombre de samples >= 1

"""
