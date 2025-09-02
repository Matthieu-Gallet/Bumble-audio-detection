#!/bin/bash

# Script pour tester différentes durées de segments et d'analyse
# Usage: ./test_durations.sh

# Configuration de base
CONFIG_FILE="scripts/config/simple_config.yaml"
BASE_CONFIG="scripts/config/simple_config_base.yaml"
PYTHON_CMD="python process_batch.py --config"

# Créer une sauvegarde du fichier de config original
if [ ! -f "$BASE_CONFIG" ]; then
    echo "Création d'une sauvegarde du fichier de configuration original..."
    cp "$CONFIG_FILE" "$BASE_CONFIG"
fi

# Liste des durées à tester
DURATIONS=(20 15 25 35 40)

echo "=========================================="
echo "Test de performance pour différentes durées"
echo "=========================================="

for duration in "${DURATIONS[@]}"; do
    echo ""
    echo "===========================================" 
    echo "Test avec durée: ${duration} secondes"
    echo "==========================================="
    
    # Modifier le fichier de configuration avec une approche plus robuste
    echo "Modification de la configuration pour durée ${duration}s..."
    
    # Utiliser sed avec des patterns plus flexibles
    cp "$BASE_CONFIG" "$CONFIG_FILE"
    
    # Modifier output_dir principal (ligne avec data_dir suivi d'output_dir)
    sed -i "/data_dir:/,/output_dir:/ s/output_dir: \"output_batch_[0-9]*\"/output_dir: \"output_batch_${duration}\"/" "$CONFIG_FILE"
    
    # Modifier segment_length
    sed -i "s/segment_length: [0-9]*/segment_length: ${duration}/" "$CONFIG_FILE"
    
    # Modifier duration dans advanced_evaluation
    sed -i "/advanced_evaluation:/,/error_analysis:/ s/duration: [0-9]*/duration: ${duration}/" "$CONFIG_FILE"
    
    # Modifier output_dir dans advanced_evaluation
    sed -i "/advanced_evaluation:/,/error_analysis:/ s|output_dir: \"output_batch_[0-9]*/advanced_evaluation\"|output_dir: \"output_batch_${duration}/advanced_evaluation\"|" "$CONFIG_FILE"
    
    # Vérifier les modifications
    echo "Configuration modifiée:"
    echo "  - output_dir principal: $(grep 'output_dir:' "$CONFIG_FILE" | head -1 | cut -d'"' -f2)"
    echo "  - segment_length: $(grep 'segment_length:' "$CONFIG_FILE" | cut -d' ' -f4)"
    echo "  - duration: $(grep -A10 'advanced_evaluation:' "$CONFIG_FILE" | grep 'duration:' | cut -d' ' -f4)"
    echo "  - advanced output_dir: $(grep -A10 'advanced_evaluation:' "$CONFIG_FILE" | grep 'output_dir:' | cut -d'"' -f2)"
    
    # Lancer le traitement
    echo "Lancement du traitement..."
    start_time=$(date +%s)
    
    if $PYTHON_CMD "$CONFIG_FILE"; then
        end_time=$(date +%s)
        duration_sec=$((end_time - start_time))
        echo "✓ Traitement terminé avec succès en ${duration_sec} secondes"
        
        # Créer un fichier de résumé pour cette durée
        mkdir -p "output_batch_${duration}"
        echo "Test durée: ${duration}s" > "output_batch_${duration}/test_summary.txt"
        echo "Temps d'exécution: ${duration_sec}s" >> "output_batch_${duration}/test_summary.txt"
        echo "Date: $(date)" >> "output_batch_${duration}/test_summary.txt"
        
    else
        echo "✗ Erreur lors du traitement pour la durée ${duration}s"
        echo "Voir les logs pour plus de détails"
    fi
    
    echo "Résultats sauvegardés dans: output_batch_${duration}/"
    echo ""
done

# Restaurer le fichier de configuration original
echo "Restauration du fichier de configuration original..."
cp "$BASE_CONFIG" "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "Tests terminés!"
echo "=========================================="
echo "Résultats disponibles dans:"
for duration in "${DURATIONS[@]}"; do
    echo "  - output_batch_${duration}/"
done

echo ""
echo "Génération d'un rapport de comparaison..."

# Créer un script de comparaison des résultats (même que précédemment)
cat > compare_results.py << 'EOF'
#!/usr/bin/env python3
import os
import json
import pandas as pd

durations = [10, 15, 20, 25, 30, 35, 40,45]
results = []

print("Comparaison des performances par durée de segment")
print("=" * 60)

for duration in durations:
    output_dir = f"output_batch_{duration}"
    
    # Chercher les fichiers de métriques dans global_advanced_evaluation
    metrics_file = None
    search_dirs = [
        os.path.join(output_dir, "global_advanced_evaluation"),
        os.path.join(output_dir, "classical_results"),
        output_dir
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith("_metrics.json"):
                        metrics_file = os.path.join(root, file)
                        break
                if metrics_file:
                    break
        if metrics_file:
            break
    
    if metrics_file and os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Extraire les métriques principales
            f1 = metrics.get('f1_score', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            roc_auc = metrics.get('roc_auc', 0)
            optimal_threshold = metrics.get('optimal_threshold', 0)
            
            results.append({
                'Duration': duration,
                'F1-Score': round(f1, 3),
                'Precision': round(precision, 3),
                'Recall': round(recall, 3),
                'ROC-AUC': round(roc_auc, 3),
                'Optimal_Threshold': round(optimal_threshold, 6)
            })
            
            print(f"Durée {duration}s: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
            
        except Exception as e:
            print(f"Erreur lors de la lecture des métriques pour {duration}s: {e}")
    else:
        print(f"Aucun fichier de métriques trouvé pour {duration}s")

if results:
    # Créer un DataFrame et sauvegarder
    df = pd.DataFrame(results)
    df.to_csv('performance_comparison.csv', index=False)
    print(f"\nTableau de comparaison:")
    print(df.to_string(index=False))
    print(f"\nRésultats sauvegardés dans: performance_comparison.csv")
    
    # Identifier la meilleure configuration
    best_f1 = df.loc[df['F1-Score'].idxmax()]
    print(f"\nMeilleure performance F1-Score:")
    print(f"  Durée: {best_f1['Duration']}s")
    print(f"  F1-Score: {best_f1['F1-Score']}")
    print(f"  Precision: {best_f1['Precision']}")
    print(f"  Recall: {best_f1['Recall']}")
else:
    print("Aucun résultat trouvé pour la comparaison")
EOF

# Rendre le script de comparaison exécutable et le lancer
chmod +x compare_results.py
echo "Lancement de l'analyse comparative..."
python compare_results.py

echo ""
echo "Script terminé. Fichiers générés:"
echo "  - performance_comparison.csv (tableau de comparaison)"
echo "  - compare_results.py (script d'analyse)"
echo "  - output_batch_XX/ (résultats pour chaque durée)"