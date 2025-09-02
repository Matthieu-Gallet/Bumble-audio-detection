# Documentation Technique - Système de Détection Acoustique

## Vue d'ensemble

Ce système permet l'analyse et l'évaluation de détections acoustiques automatisées, avec support pour l'optimisation de seuils, l'analyse d'erreurs par classe, et l'évaluation à différentes échelles (locale et globale).

## Configuration de l'environnement

### Création de l'environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Activation de l'environnement virtuel

```bash
source .venv/bin/activate
```

## Structure des données

### Entrées

1. **Fichiers audio** : Format WAV, segments de 10 secondes
2. **Annotations** : Fichiers texte (.txt) avec format :
   ```
   start_time    end_time    label
   12.5          22.5        buzz
   ```
3. **Prédictions** : Fichier CSV avec colonnes de scores pour chaque classe
4. **Configuration** : Fichier YAML avec paramètres d'analyse

### Sorties

1. **Résultats de détection** : Fichiers CSV avec scores par segment
2. **Métriques d'évaluation** : Fichiers JSON et TXT
3. **Visualisations** : Graphiques PNG (courbes ROC, matrices de confusion, analyses d'erreurs)
4. **Résultats agrégés** : Fichier merged_results.csv consolidant tous les sites

## Commandes principales

### Traitement par lots

```bash
python process_batch.py --config scripts/config/simple_config.yaml
```


## Taxonomie des classes

### Classes primaires (TaggingCategory.csv)

- **fly_housefly** : Mouches, mouches domestiques
- **bee_wasp** : Abeilles, guêpes
- **water** : Sons d'eau (ruisseau, cascade, pluie)
- **wind** : Sons de vent (feuilles, bruit de micro)
- **motor_vehicle** : Véhicules motorisés (voiture, klaxon, freinage)
- **aircraft** : Aéronefs (avion, hélicoptère, moteur à réaction)
- **human_voice** : Voix humaine (parole, chant, rire, pleurs)

### Meta-classes (configuration)

- **Group_buzz** : Regroupement des sons d'insectes bourdonnants
- **Group_geophony** : Sons naturels non-biologiques (eau, vent)
- **Group_anthropophony** : Sons d'origine humaine (véhicules, voix)



## Métriques d'évaluation

### F1-Score
Mesure combinant précision et rappel : F1 = 2 × (Précision × Rappel) / (Précision + Rappel)

### Courbe ROC (Receiver Operating Characteristic)
Graphique représentant le taux de vrais positifs vs le taux de faux positifs pour différents seuils de décision.

### Matrice de confusion
Tableau croisé des prédictions vs vérité terrain :
- **Vrais Positifs (TP)** : Sons correctement détectés
- **Faux Positifs (FP)** : Sons détectés à tort (fausses alarmes)
- **Vrais Négatifs (TN)** : Absences correctement identifiées  
- **Faux Négatifs (FN)** : Sons manqués par le système

**Faux Positifs** : Le système signale la présence d'un son cible alors qu'il n'y en a pas. Ces erreurs génèrent des fausses alarmes.

**Faux Négatifs** : Le système ne détecte pas un son cible qui est pourtant présent. Ces erreurs correspondent à des détections manquées.

## Classe "None" dans l'analyse d'erreurs

### Définition

La classe "none" représente les cas où aucune classe ne dépasse le seuil de détection configuré (`used_threshold`).

### Utilisation

1. **Faux positifs "none"** : Le système détecte une classe cible mais aucune autre classe ne dépasse le seuil
2. **Faux négatifs "none"** : Le système rate une détection et aucune classe n'atteint le seuil

### Interprétation

- **Forte présence de "none"** : Indique des scores globalement faibles
- **Faible présence de "none"** : Indique des confusions actives entre classes spécifiques

### Algorithme de classification des erreurs

```python
for segment in error_segments:
    classes_above_threshold = [classe for classe in available_columns 
                              if score[classe] > used_threshold]
    if not classes_above_threshold:
        assigned_class = "none"
    else:
        assigned_class = max(classes_above_threshold, key=lambda x: score[x])
```

## Optimisation de seuil

### Méthode

Recherche du seuil optimal maximisant le F1-Score sur une grille de 200 points entre 0.001 et 1.0.

### Algorithme

1. **Génération de seuils** : `np.linspace(0.001, 1.0, 200)`
2. **Évaluation** : Calcul du F1-Score pour chaque seuil
3. **Sélection** : Choix du seuil avec le F1-Score maximal
4. **Application** : Réévaluation avec le seuil optimal

### Métriques de sortie

- **optimal_threshold** : Seuil optimal identifié
- **optimal_f1** : F1-Score correspondant
- **threshold_results** : Courbe complète seuil/F1-Score

## Analyse locale vs globale

### Analyse locale (site-level)

- **Périmètre** : Évaluation par site/session individuellement
- **Données** : Annotations et prédictions d'un site unique
- **Sortie** : Métriques spécifiques à chaque site
- **Répertoire** : `output_batch/{site_name}/`

### Analyse globale (global)

- **Périmètre** : Évaluation sur l'ensemble des sites
- **Données** : Agrégation de tous les fichiers dans merged_results.csv
- **Sortie** : Métriques consolidées cross-sites
- **Répertoire** : `output_batch/classical_results/` et `output_batch/global_advanced_evaluation/`

### Configuration de portée

Parameter `evaluation_scope` dans simple_config.yaml :
- **"local"** : Analyse par site uniquement
- **"global"** : Analyse globale uniquement  
- **"both"** : Analyses locale et globale

## Configuration avancée

### Exclusion de classes

Nécessaire car des classes sont imbriquées

```yaml
advanced_evaluation:
  error_analysis:
    excluded_classes:
      Group_buzz:
        - "tag_bee_wasp"
        - "tag_fly_housefly"
        - "Group_buzz"
```

## Configuration (simple_config.yaml)

### Chemins et environnement
```yaml
paths:
  data_dir: "data2"              # Répertoire des données audio
  output_dir: "output_batch"     # Répertoire de sortie
  python_path: ".venv/bin/python"
  process_script: "process.py"
```

### Traitement
```yaml
processing:
  segment_length: 10             # Durée des segments audio (secondes)
  audio_format: "wav"            # Format audio traité
  timeout: 1800                  # Timeout de traitement (secondes)
```

### Modèle
```yaml
model:
  type: "mobilenetv2"            # Type de modèle (mobilenetv2, resnet22)
```

### Analyse
```yaml
analysis:
  mode: "evaluation"             # Mode : "inference" ou "evaluation"
  evaluation_scope: "both"       # Portée : "local", "global" ou "both"
  
  ground_truth:
    annotations_dir: "data"      # Répertoire des annotations
    annotation_pattern: "*_annotées"  # Motif des dossiers d'annotations
  
  evaluation:
    results_subdir: "results"
    columns: ["Group_buzz"]      # Colonnes à évaluer
    default_threshold: 0.2       # Seuil par défaut
```

### Évaluation avancée
```yaml
advanced_evaluation:
  enabled: true                  # Activation de l'évaluation avancée
  optimize_threshold: true       # Optimisation automatique du seuil
  duration: 10                   # Durée pour analyse temporelle
  output_dir: "output_batch/advanced_evaluation"
  
  error_analysis:
    excluded_classes:            # Classes exclues de l'analyse d'erreurs
      Group_buzz:
        - "tag_bee_wasp"         # Éviter double comptage
        - "tag_fly_housefly"
        - "Group_buzz"
```

### Paramètres temporels

- **segment_length** : 10 secondes (durée des segments d'analyse)
- **duration** : 10 secondes (durée pour l'analyse temporelle)
- **tolerance** : 2.5 secondes (tolérance d'alignement temporal)

## Structure de sortie

```
output_batch/
├── merged_results.csv                    # Agrégation globale
├── {site_name}/
│   ├── classical_evaluation/
│   │   └── eval_{column}/
│   │       ├── detailed_results.csv
│   │       ├── metrics.txt
│   │       └── confusion_matrix.png
│   └── advanced_evaluation/
│       └── advanced_{column}/
│           ├── advanced_results_{column}.csv
│           ├── advanced_metrics_{column}.json
│           ├── advanced_analysis_{column}.png
│           └── error_analysis_{column}.png
├── classical_results/                   # Résultats globaux classiques
└── global_advanced_evaluation/          # Résultats globaux avancés
```

## Analyses
Le script

## TODO

enregistreurs : Loriaz 1600 (voix) / Loriaz 2100 (vent) / Peclerey 1400 (Helico) / Diosaz (riviere)
- période : 1er avril → 20 juin. toute la journée (si pas trop long...)
