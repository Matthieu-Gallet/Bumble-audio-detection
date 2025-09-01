#!/usr/bin/env python3
"""
Script d'évaluation avancée autonome avec interface CLI.

Ce script permet de lancer une évaluation avancée complète incluant:
- Optimisation du seuil par F1-score
- Métriques avancées (weighted F1, ROC-AUC, PR-AUC)
- Visualisations complètes (ROC, PR, distributions, etc.)
- Analyse des erreurs

Usage:
    python advanced_eval_cli.py --csv path/to/predictions.csv --annotations path/to/annotations/ --column Group_buzz --output path/to/output/
"""

import argparse
import os
import sys

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from evaluate_detection import run_advanced_evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Évaluation avancée des détections audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    # Évaluation avec optimisation du seuil
    python advanced_eval_cli.py --csv output_batch/merged_results.csv --annotations data/session_01/SM05_T_annotées --column Group_buzz --output output_batch/evaluation_advanced
    
    # Évaluation sans optimisation du seuil
    python advanced_eval_cli.py --csv output_batch/merged_results.csv --annotations data/session_01/SM05_T_annotées --column Group_buzz --output output_batch/evaluation_advanced --no-optimize
    
    # Segments de 5 secondes au lieu de 10
    python advanced_eval_cli.py --csv output_batch/merged_results.csv --annotations data/session_01/SM05_T_annotées --column Group_buzz --output output_batch/evaluation_advanced --duration 5.0
        """,
    )

    parser.add_argument(
        "--csv", required=True, help="Chemin vers le fichier CSV avec les prédictions"
    )

    parser.add_argument(
        "--annotations",
        required=True,
        help="Chemin vers le répertoire avec les annotations de vérité terrain",
    )

    parser.add_argument(
        "--column",
        required=True,
        help="Nom de la colonne à évaluer (ex: Group_buzz, tag_Buzz, etc.)",
    )

    parser.add_argument(
        "--output", required=True, help="Répertoire de sortie pour les résultats"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Durée des segments en secondes (défaut: 10.0)",
    )

    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Désactiver l'optimisation automatique du seuil",
    )

    args = parser.parse_args()

    # Validation des arguments
    if not os.path.exists(args.csv):
        print(f"❌ Erreur: Le fichier CSV '{args.csv}' n'existe pas")
        sys.exit(1)

    if not os.path.exists(args.annotations):
        print(
            f"❌ Erreur: Le répertoire d'annotations '{args.annotations}' n'existe pas"
        )
        sys.exit(1)

    # Affichage de la configuration
    print("🔧 CONFIGURATION")
    print("=" * 50)
    print(f"📁 CSV: {args.csv}")
    print(f"📋 Annotations: {args.annotations}")
    print(f"📊 Colonne: {args.column}")
    print(f"📂 Sortie: {args.output}")
    print(f"⏱️  Durée segments: {args.duration}s")
    print(f"🎯 Optimisation seuil: {'Non' if args.no_optimize else 'Oui'}")

    try:
        # Lancement de l'évaluation avancée
        results = run_advanced_evaluation(
            csv_path=args.csv,
            annotations_dir=args.annotations,
            detection_column=args.column,
            output_dir=args.output,
            duration=args.duration,
            optimize_threshold=not args.no_optimize,
        )

        # Résumé final
        print("\n" + "=" * 60)
        print("🎉 ÉVALUATION TERMINÉE AVEC SUCCÈS!")
        print("=" * 60)

        metrics = results["metrics"]
        print(f"📊 Segments analysés: {len(results['segments_df'])}")
        print(f"🎯 Seuil final: {results['threshold_used']:.6f}")

        print(f"\n📈 MÉTRIQUES PRINCIPALES:")
        print("-" * 30)
        print(f"Précision:    {metrics.get('precision', 0):.3f}")
        print(f"Rappel:       {metrics.get('recall', 0):.3f}")
        print(f"F1-Score:     {metrics.get('f1_score', 0):.3f}")
        print(f"Weighted F1:  {metrics.get('weighted_f1', 0):.3f}")
        print(f"Accuracy:     {metrics.get('accuracy', 0):.3f}")
        print(f"ROC-AUC:      {metrics.get('roc_auc', 0):.3f}")

        if results.get("optimization_results"):
            opt = results["optimization_results"]
            print(f"\n🎯 OPTIMISATION DU SEUIL:")
            print("-" * 30)
            print(f"F1 optimal:    {opt['optimal_f1']:.3f}")
            print(f"Seuil optimal: {opt['optimal_threshold']:.6f}")

        print(f"\n📁 Fichiers générés dans: {args.output}")
        print(f"   📊 Graphiques: advanced_analysis_{args.column}.png")
        print(f"   📋 Résultats détaillés: advanced_results_{args.column}.csv")
        print(f"   📈 Métriques JSON: advanced_metrics_{args.column}.json")

    except Exception as e:
        print(f"\n❌ ERREUR lors de l'évaluation:")
        print(f"   {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
