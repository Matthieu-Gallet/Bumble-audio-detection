# Rapport d'analyses - Détection acoustique automatisée

Ce document présente les résultats de trois analyses distinctes réalisées sur des données acoustiques en utilisant un modèle de détection automatique. Chaque analyse répond à des objectifs spécifiques avec des configurations et des données différentes.

## 1. Analyse temporelle - Optimisation de la taille de fenêtre

### Objectif
Déterminer la taille de fenêtre d'analyse optimale pour la détection de bourdonnements d'insectes en évaluant les performances du modèle sur des fenêtres de 5 à 45 secondes.

### Configuration technique
- **Modèle** : MobileNetV2
- **Mode d'analyse** : Evaluation avec vérité terrain
- **Données d'entrée** : Données labelisées issues du dossier `data_org/`
  - 4 sessions terrain (20240408, 20240612, 20240724, 20240807)
  - Fichiers audio annotés manuellement (fichiers .txt dans dossiers `*_annotées`)
- **Fenêtres testées** : 5, 10, 15, 20, 25, 30, 35, 40, 45 secondes
- **Métrique cible** : Group_buzz (détection d'insectes bourdonnants)

### Paramètres d'évaluation
- **Optimisation du seuil** : Activée (recherche du seuil optimal par analyse ROC)
- **Scope d'évaluation** : Global (agrégation de toutes les sessions)
- **Configuration** : Itération sur `scripts/config/simple_config.yaml` avec modification dynamique du paramètre `segment_length`

### Résultats principaux

#### Performance par taille de fenêtre
| Fenêtre (s) | F1-Score Optimal | Seuil Optimal | ROC-AUC | Précision | Rappel |
|-------------|------------------|---------------|---------|-----------|---------|
| 5           | 0.568           | 0.152         | 0.778   | 0.740     | 0.461   |
| 10          | 0.660           | 0.202         | 0.851   | 0.729     | 0.602   |
| 15          | 0.670           | 0.273         | 0.864   | 0.702     | 0.640   |
| 20          | 0.676           | 0.242         | 0.874   | 0.666     | 0.686   |
| 25          | 0.680           | 0.424         | 0.881   | 0.696     | 0.664   |
| **30**      | **0.683**       | 0.354         | 0.887   | 0.657     | 0.711   |
| 35          | 0.678           | 0.394         | **0.891** | 0.642   | 0.714   |
| 40          | 0.660           | 0.354         | 0.887   | 0.604     | 0.728   |
| 45          | 0.650           | 0.535         | 0.888   | 0.638     | 0.663   |

#### Métriques optimales identifiées
- **Meilleur F1-Score optimal** : 0.683 (fenêtre de 30 secondes)
- **Meilleur ROC-AUC** : 0.891 (fenêtre de 35 secondes)
- **Tendance générale** : Performance croissante jusqu'à 30-35 secondes, puis décroissance

### Interprétation
Le F1-Score optimisé correspond à la maximisation du score F1 en ajustant le seuil de classification. Cette optimisation permet de trouver l'équilibre optimal entre précision et rappel pour chaque taille de fenêtre. Les résultats montrent qu'une fenêtre de 30 secondes offre le meilleur compromis pour la détection des bourdonnements d'insectes.

## 2. Analyse d'inférence - Détection sur données réelles

### Objectif
Appliquer le modèle de détection sur des données acoustiques réelles non-labelisées pour analyser les patterns de détection temporels et spatiaux.

### Configuration technique
- **Modèle** : MobileNetV2
- **Mode d'analyse** : Inference (sans vérité terrain)
- **Fenêtre d'analyse** : 30 secondes (basé sur l'analyse temporelle)
- **Configuration** : `scripts/config/simple_config2.yaml`

### Données d'entrée
Données sélectionnées via le script `select_acoustic_data.py` selon les critères suivants :
- **Sites** : 4 enregistreurs acoustiques
  - D01 (SMA02939) : Diosaz - Servoz, altitude 838m
  - D04 (SMA02961) : Peclerey, altitude 1400m  
  - D05 (SMA02964) : Loriaz 1600, altitude 1630m
  - D08 (SMA02975) : Loriaz 2100, altitude 2140m
- **Période temporelle** : 1er avril au 20 juin 2025
- **Heures d'enregistrement** : 8h00 à 21h00
- **Source** : `/mnt/BACK UP/select_data`

### Paramètres d'analyse
- **Groupes analysés** : 
  - Group_buzz (Insectes) - seuils : 0.3, 0.485
  - Group_anthropophony (Activité humaine) - seuils : 0.3, 0.5
  - Group_geophony (Sons naturels) - seuils : 0.3, 0.5
- **Segmentation temporelle** : 30 secondes
- **Format audio** : WAV
- **Traitement** : Multiprocessing (10 processus), batch size 32

### Résultats principaux

#### Statistiques de détection Group_buzz (seuil 0.3)
| Site | Altitude | Avril | Mai | Juin | Taux détection global |
|------|----------|-------|-----|------|----------------------|
| D04  | 1400m    | 0.66% | 1.11% | 0.40% | 0.72% |
| D05  | 1630m    | 0.07% | 0.57% | 0.69% | 0.44% |
| D08  | 2140m    | -     | 6.60% | 18.35% | 12.48% |

#### Patterns temporels identifiés
- **Variation altitudinale** : Augmentation significative des détections avec l'altitude (D08 > D05 > D04)
- **Évolution saisonnière** : Progression des détections d'avril à juin, particulièrement marquée sur le site D08
- **Patterns horaires** : Pics de détection variables selon les sites et les mois

### Types de résultats générés
- **Séries temporelles** : Évolution des probabilités de détection par site
- **Heatmaps horaires** : Taux de détection par heure et par mois
- **Statistiques descriptives** : Moyennes, médianes, maxima des probabilités par groupe

## 3. Analyse spatiale - Évaluation comparative par site

### Objectif
Évaluer les performances du modèle en mode évaluation complète avec analyse par site et globalisation des résultats.

### Configuration technique
- **Modèle** : MobileNetV2  
- **Mode d'analyse** : Evaluation (avec vérité terrain)
- **Scope d'évaluation** : Both (analyse par site ET analyse globale)
- **Fenêtre d'analyse** : 30 secondes
- **Configuration** : `scripts/config/simple_config.yaml`

### Données d'entrée
Données labelisées du dossier `data_org/` :
- **Sessions terrain 2024** :
  - 20240408_session_01_Tent (SM05_T)
  - 20240612_session_02_Tent (SM06_T)  
  - 20240724_session_03_Tent (SM06_T)
  - 20240807_session_04_Tent (SM03_T)
- **Stations pollisophenocatch** :
  - SM02_Colin
  - SM02_Jeremy  
  - SM04_Enola
- **Annotations** : Fichiers .txt dans dossiers `*_annotées`

### Paramètres d'évaluation
- **Métrique principale** : Group_buzz
- **Optimisation du seuil** : Activée (maximisation du F1-Score)
- **Seuil par défaut** : 0.2
- **Analyses** :
  - Classical evaluation (seuil fixe)
  - Advanced evaluation (seuil optimisé)
  - Error analysis (analyse des faux positifs/négatifs)

### Métriques d'étude

#### F1-Score optimisé
Le F1-Score optimisé correspond à la maximisation du score F1 en ajustant dynamiquement le seuil de classification. Cette approche permet de :
- Trouver l'équilibre optimal entre précision et rappel
- Adapter le seuil aux caractéristiques spécifiques des données
- Maximiser la performance globale de détection

#### Métriques calculées
- **Métriques de base** : Précision, Rappel, F1-Score, Accuracy
- **Métriques avancées** : Spécificité, Sensitivité, ROC-AUC, PR-AUC
- **Métriques pondérées** : Weighted precision/recall/F1 pour classes déséquilibrées
- **Métriques macro** : Moyennes non-pondérées pour évaluation équitable

### Résultats principaux

#### Performance globale (analyse avancée)
- **F1-Score** : 0.532 (53.2%)
- **F1-Score optimisé** : 0.658 (seuil optimal : 0.354)
- **Précision** : 0.451
- **Rappel** : 0.651
- **ROC-AUC** : 0.877
- **Accuracy** : 0.917

#### Comparaison par session
Les sessions terrain 2024 montrent des performances variables avec des améliorations significatives en mode advanced (seuil optimisé) par rapport au mode classical (seuil fixe). Mais grosses disparités entre les sessions. **Attention** : l'analyse de pollisophenocatch montre des performances très faibles mais en prenant toute la piste audio (et pas seulement les segments annotés).

#### Matrices de confusion
- **Vrais positifs** : 803
- **Faux positifs** : 979  
- **Vrais négatifs** : 14,774
- **Faux négatifs** : 431

### Types de résultats générés
- **Évaluation classique** : Métriques avec seuil fixe par site
- **Évaluation avancée** : Métriques avec seuil optimisé par site et global
- **Analyse d'erreurs** : Identification et catégorisation des erreurs de classification
- **Comparaisons visuelles** : Courbes ROC, Precision-Recall, matrices de confusion

## Conclusions générales

Les trois analyses révèlent des aspects complémentaires du système de détection :

1. **Optimisation temporelle** : Fenêtre de 30 secondes identifiée comme optimale
2. **Application réelle** : Validation sur données non-labelisées avec patterns écologiques cohérents  
3. **Évaluation spatiale** : Performance satisfaisante avec marge d'amélioration via optimisation des seuils

L'ensemble confirme la viabilité du modèle MobileNetV2 pour la détection automatique de bourdonnements d'insectes avec des performances variables selon les conditions d'enregistrement et les paramètres d'analyse.
