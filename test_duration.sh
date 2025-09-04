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
DURATIONS=(5 10 15 20 25 30 35 40 45)

# Vérifier que le répertoire de données source existe
if [ ! -d "data" ]; then
    echo "Erreur: Le répertoire 'data' n'existe pas!"
    echo "Assurez-vous que les données sources sont disponibles dans le répertoire 'data'."
    exit 1
fi

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
    
    # Copier la configuration de base et appliquer les modifications
    cp "$BASE_CONFIG" "$CONFIG_FILE"
    
    # Modifier data_dir pour utiliser le bon répertoire source
    sed -i 's|data_dir: "output_final_30"|data_dir: "data"|' "$CONFIG_FILE"
    sed -i 's|data_dir: "data"|data_dir: "data"|' "$CONFIG_FILE"  # S'assurer que c'est bien data
    
    # Modifier output_dir principal 
    sed -i 's|output_dir: "output_final_30"|output_dir: "output_batch_'${duration}'"|' "$CONFIG_FILE"
    sed -i "s|output_dir: \"output_batch_[0-9]*\"|output_dir: \"output_batch_${duration}\"|" "$CONFIG_FILE"
    
    # Modifier segment_length
    sed -i "s/segment_length: [0-9]*/segment_length: ${duration}/" "$CONFIG_FILE"
    
    # Modifier duration dans advanced_evaluation
    sed -i "/advanced_evaluation:/,/error_analysis:/ s/duration: [0-9]*/duration: ${duration}/" "$CONFIG_FILE"
    
    # Modifier output_dir dans advanced_evaluation
    sed -i "/advanced_evaluation:/,/error_analysis:/ s|output_dir: \"output_batch_[0-9]*/advanced_evaluation\"|output_dir: \"output_batch_${duration}/advanced_evaluation\"|" "$CONFIG_FILE"
    
    # Créer le répertoire de sortie s'il n'existe pas
    mkdir -p "output_batch_${duration}"
    
    # Vérifier les modifications
    echo "Configuration modifiée:"
    echo "  - data_dir: $(grep 'data_dir:' "$CONFIG_FILE" | cut -d'"' -f2)"
    echo "  - output_dir principal: $(grep 'output_dir:' "$CONFIG_FILE" | head -1 | cut -d'"' -f2)"
    echo "  - segment_length: $(grep 'segment_length:' "$CONFIG_FILE" | awk '{print $2}')"
    echo "  - duration: $(grep -A10 'advanced_evaluation:' "$CONFIG_FILE" | grep 'duration:' | awk '{print $2}')"
    echo "  - advanced output_dir: $(grep -A10 'advanced_evaluation:' "$CONFIG_FILE" | grep 'output_dir:' | cut -d'"' -f2)"
    echo "  - segment_length: $(grep 'segment_length:' "$CONFIG_FILE" | cut -d' ' -f4)"
    echo "  - duration: $(grep -A10 'advanced_evaluation:' "$CONFIG_FILE" | grep 'duration:' | cut -d' ' -f4)"
    echo "  - advanced output_dir: $(grep -A10 'advanced_evaluation:' "$CONFIG_FILE" | grep 'output_dir:' | cut -d'"' -f2)"
    
    # Lancer le traitement
    echo "Lancement du traitement..."
    start_time=$(date +%s)
    
    # Activer l'environnement virtuel si nécessaire
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    if $PYTHON_CMD "$CONFIG_FILE"; then
        end_time=$(date +%s)
        duration_sec=$((end_time - start_time))
        echo "✓ Traitement terminé avec succès en ${duration_sec} secondes"
        
        # Créer un fichier de résumé pour cette durée
        echo "Test durée: ${duration}s" > "output_batch_${duration}/test_summary.txt"
        echo "Temps d'exécution: ${duration_sec}s" >> "output_batch_${duration}/test_summary.txt"
        echo "Date: $(date)" >> "output_batch_${duration}/test_summary.txt"
        echo "Segment length: ${duration}s" >> "output_batch_${duration}/test_summary.txt"
        echo "Data source: data/" >> "output_batch_${duration}/test_summary.txt"
        
        # Vérifier que des fichiers ont été générés
        if [ -d "output_batch_${duration}" ] && [ "$(ls -A output_batch_${duration} 2>/dev/null | wc -l)" -gt 1 ]; then
            echo "✓ Fichiers de résultats générés dans output_batch_${duration}/"
            echo "  Contenu du répertoire:"
            ls -la "output_batch_${duration}/" | head -10
        else
            echo "⚠ Attention: Peu ou pas de fichiers générés dans output_batch_${duration}/"
        fi
        
    else
        echo "✗ Erreur lors du traitement pour la durée ${duration}s"
        echo "Voir les logs pour plus de détails"
        
        # Créer quand même un fichier de résumé pour l'erreur
        mkdir -p "output_batch_${duration}"
        echo "Test durée: ${duration}s - ÉCHEC" > "output_batch_${duration}/test_summary.txt"
        echo "Erreur lors de l'exécution" >> "output_batch_${duration}/test_summary.txt"
        echo "Date: $(date)" >> "output_batch_${duration}/test_summary.txt"
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
python3 temporal_analysis_run_advanced.py
python3 temporal_analysis_plot_metrics.py