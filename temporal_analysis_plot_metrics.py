#!/usr/bin/env python3
"""
Script pour analyser et visualiser les métriques de performance en fonction de la taille de la fenêtre d'analyse.
Récupère les données des fichiers advanced_metrics_Group_buzz.json dans tous les dossiers output_batch_XX.
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def find_metrics_files():
    """Trouve tous les fichiers de métriques avancées dans les dossiers output_batch_XX."""
    base_dir = "/home/mgallet/Documents/Acoustique/detection"
    metrics_data = []
    
    # Chercher tous les dossiers output_batch_XX
    output_dirs = [
        d
        for d in os.listdir(base_dir)
        if d.startswith("output_batch_") and os.path.isdir(os.path.join(base_dir, d))
    ]
    output_dirs.sort(key=lambda x: int(x.split("_")[-1]))  # Trier par taille de fenêtre

    print(f"Dossiers trouvés: {output_dirs}")

    for output_dir in output_dirs:
        # Extraire la taille de fenêtre du nom du dossier
        window_size = int(output_dir.split("_")[-1])

        # Chercher le fichier de métriques avancées dans le dossier global
        pattern = os.path.join(
            base_dir,
            output_dir,
            "global_global_advanced_evaluation/advanced_Group_buzz/advanced_metrics_Group_buzz.json",
        )
        metrics_files = glob.glob(pattern, recursive=False)

        if metrics_files:
            metrics_file = metrics_files[0]  # Prendre le premier trouvé
            print(f"Fenêtre {window_size}s: {metrics_file}")

            try:
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)

                # Extraire les métriques principales
                data = {
                    "window_size": window_size,
                    "f1_score": metrics.get("f1_score", None),
                    "optimal_f1": metrics.get("optimal_f1", None),
                    "precision": metrics.get(
                        "precision", metrics.get("Precision", None)
                    ),
                    "recall": metrics.get("recall", metrics.get("Recall", None)),
                    "roc_auc": metrics.get("roc_auc", metrics.get("ROC-AUC", None)),
                    "optimal_threshold": metrics.get(
                        "optimal_threshold", metrics.get("Optimal_Threshold", None)
                    ),
                    "accuracy": metrics.get("accuracy", metrics.get("Accuracy", None)),
                    "pr_auc": metrics.get("pr_auc", metrics.get("PR-AUC", None)),
                }

                metrics_data.append(data)
                print(
                    f"  ✓ F1: {data['f1_score']:.3f}, Optimal F1: {data['optimal_f1']:.3f}, Seuil optimal: {data['optimal_threshold']:.6f}, ROC-AUC: {data['roc_auc']:.3f}"
                )

            except Exception as e:
                print(f"  ✗ Erreur lecture {metrics_file}: {e}")
        else:
            print(f"Fenêtre {window_size}s: Aucun fichier de métriques trouvé")

    return metrics_data


def create_plots(metrics_data, output_dir):
    """Crée les graphiques de visualisation des métriques."""
    if not metrics_data:
        print("Aucune donnée de métriques trouvée")
        return

    # Convertir en DataFrame pour faciliter la manipulation
    df = pd.DataFrame(metrics_data)
    df = df.sort_values("window_size")

    print(f"\nDonnées collectées pour {len(df)} tailles de fenêtre:")
    print(
        df[["window_size", "optimal_f1", "optimal_threshold", "roc_auc"]].to_string(
            index=False
        )
    )

    # Créer la figure avec 3 sous-graphiques verticaux (3x1)
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    fig.suptitle(
        "Analyse des métriques en fonction de la taille de la fenêtre d'analyse",
        fontsize=16,
        fontweight="bold",
    )

    # Couleurs pour les graphiques
    colors = ["#A23B72", "#F18F01", "#00A8A8"]

    # 1. F1-Score Optimal
    ax1 = axes[0]
    ax1.plot(
        df["window_size"],
        df["optimal_f1"],
        "o-",
        color=colors[0],
        linewidth=2,
        markersize=8,
    )
    ax1.set_xlabel("Taille de la fenêtre (secondes)", fontsize=12)
    ax1.set_ylabel("F1-Score Optimal", fontsize=12)
    ax1.set_title("Évolution du F1-Score Optimal", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(df["optimal_f1"]) * 1.1)

    # Ajouter les valeurs sur les points
    for i, (x, y) in enumerate(zip(df["window_size"], df["optimal_f1"])):
        ax1.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    # Identifier le meilleur F1-Score Optimal
    best_opt_f1_idx = df["optimal_f1"].idxmax()
    best_opt_f1_window = df.loc[best_opt_f1_idx, "window_size"]
    best_opt_f1_score = df.loc[best_opt_f1_idx, "optimal_f1"]
    ax1.scatter(best_opt_f1_window, best_opt_f1_score, color="red", s=100, zorder=5)
    ax1.annotate(
        f"Meilleur: {best_opt_f1_window}s",
        xy=(best_opt_f1_window, best_opt_f1_score),
        xytext=(0, -15),  # Position 1.5x plus bas
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        fontsize=10,
        fontweight="bold",
        ha="center",
    )

    # 2. Seuil optimal
    ax2 = axes[1]
    ax2.plot(
        df["window_size"],
        df["optimal_threshold"],
        "s-",
        color=colors[1],
        linewidth=2,
        markersize=8,
    )
    ax2.set_xlabel("Taille de la fenêtre (secondes)", fontsize=12)
    ax2.set_ylabel("Seuil optimal", fontsize=12)
    ax2.set_title("Évolution du seuil optimal", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Ajouter les valeurs sur les points
    for i, (x, y) in enumerate(zip(df["window_size"], df["optimal_threshold"])):
        ax2.annotate(
            f"{y:.4f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    # 3. ROC-AUC
    ax3 = axes[2]
    ax3.plot(
        df["window_size"],
        df["roc_auc"],
        "^-",
        color=colors[2],
        linewidth=2,
        markersize=8,
    )
    ax3.set_xlabel("Taille de la fenêtre (secondes)", fontsize=12)
    ax3.set_ylabel("ROC-AUC", fontsize=12)
    ax3.set_title("Évolution du ROC-AUC", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    # Ajouter les valeurs sur les points
    for i, (x, y) in enumerate(zip(df["window_size"], df["roc_auc"])):
        ax3.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    # Identifier le meilleur ROC-AUC
    best_roc_idx = df["roc_auc"].idxmax()
    best_roc_window = df.loc[best_roc_idx, "window_size"]
    best_roc_score = df.loc[best_roc_idx, "roc_auc"]
    ax3.scatter(best_roc_window, best_roc_score, color="red", s=100, zorder=5)
    ax3.annotate(
        f"Meilleur: {best_roc_window}s",
        xy=(best_roc_window, best_roc_score),
        xytext=(0, -15),  # Position 1.5x plus bas
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        fontsize=10,
        fontweight="bold",
        ha="center",
    )

    # Ajuster la mise en page
    plt.tight_layout()

    # Sauvegarder le graphique
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "metrics_analysis_by_window_size.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nGraphique sauvegardé: {output_file}")

    # Sauvegarder aussi les données en CSV
    csv_file = os.path.join(output_dir, "metrics_data_by_window_size.csv")
    df.to_csv(csv_file, index=False)
    print(f"Données sauvegardées: {csv_file}")

    # Afficher un résumé des résultats
    print(f"\n{'='*60}")
    print("RÉSUMÉ DES RÉSULTATS")
    print(f"{'='*60}")
    
    # Afficher les résultats
    print(f"\nMeilleur F1-Score Optimal: {best_opt_f1_score:.3f} (fenêtre: {best_opt_f1_window}s)")
    print(f"Meilleur ROC-AUC: {best_roc_score:.3f} (fenêtre: {best_roc_window}s)")

    # Analyse des tendances (seulement pour les métriques affichées)
    opt_f1_trend = "croissante" if df["optimal_f1"].iloc[-1] > df["optimal_f1"].iloc[0] else "décroissante"
    roc_trend = "croissante" if df["roc_auc"].iloc[-1] > df["roc_auc"].iloc[0] else "décroissante"
    threshold_trend = "croissante" if df["optimal_threshold"].iloc[-1] > df["optimal_threshold"].iloc[0] else "décroissante"

    print(f"\nTendances générales:")
    print(f"  F1-Score Optimal: {opt_f1_trend}")
    print(f"  ROC-AUC: {roc_trend}")
    print(f"  Seuil optimal: {threshold_trend}")

    return output_file


def main():
    """Fonction principale."""
    print("Analyse des métriques de performance par taille de fenêtre")
    print("=" * 60)

    # Collecter les données de métriques
    metrics_data = find_metrics_files()

    if not metrics_data:
        print("Aucune donnée de métriques trouvée. Vérifiez que les évaluations avancées ont été exécutées.")
        return

    # Créer les graphiques
    analysis_dir = "/home/mgallet/Documents/Acoustique/detection/analysis"
    output_file = create_plots(metrics_data, analysis_dir)

    print(f"\nAnalyse terminée. Consultez le fichier: {output_file}")


if __name__ == "__main__":
    main()
