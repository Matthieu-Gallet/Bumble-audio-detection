#!/usr/bin/env python3
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

durations = [10, 15, 20, 25, 30, 35, 40, 45]
results = []

print("Comparaison des performances par durée de segment")
print("=" * 60)

for duration in durations:
    output_dir = f"output_batch_{duration}"

    # Chercher les fichiers de métriques dans global_advanced_evaluation
    metrics_file = None
    search_paths = [
        os.path.join(
            output_dir,
            "global_advanced_evaluation",
            "advanced_Group_buzz",
            "advanced_metrics_Group_buzz.json",
        ),
        os.path.join(output_dir, "global_advanced_evaluation"),
        os.path.join(output_dir, "classical_results"),
        output_dir,
    ]

    # Chercher d'abord le fichier spécifique
    for search_path in search_paths:
        if os.path.isfile(search_path):
            metrics_file = search_path
            break
        elif os.path.isdir(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith("_metrics.json") or "advanced_metrics" in file:
                        metrics_file = os.path.join(root, file)
                        break
                if metrics_file:
                    break
            if metrics_file:
                break

    if metrics_file and os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            # Extraire les métriques principales
            f1 = metrics.get("f1_score", 0)
            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)
            roc_auc = metrics.get("roc_auc", 0)
            optimal_threshold = metrics.get("optimal_threshold", 0)

            results.append(
                {
                    "Duration": duration,
                    "F1-Score": round(f1, 3),
                    "F1-Score_%": round(f1 * 100, 1),  # F1-Score en pourcentage
                    "Precision": round(precision, 3),
                    "Recall": round(recall, 3),
                    "ROC-AUC": round(roc_auc, 3),
                    "Optimal_Threshold": round(optimal_threshold, 6),
                }
            )

            print(
                f"Durée {duration}s: F1={f1:.3f} ({f1*100:.1f}%), Precision={precision:.3f}, Recall={recall:.3f}"
            )

        except Exception as e:
            print(f"Erreur lors de la lecture des métriques pour {duration}s: {e}")
    else:
        print(
            f"Aucun fichier de métriques trouvé pour {duration}s (cherché dans {search_paths[0]})"
        )

if results:
    # Créer un DataFrame et sauvegarder
    df = pd.DataFrame(results)
    df.to_csv("performance_comparison.csv", index=False)
    print(f"\nTableau de comparaison:")
    print(df.to_string(index=False))
    print(f"\nRésultats sauvegardés dans: performance_comparison.csv")

    # Identifier la meilleure configuration
    best_f1 = df.loc[df["F1-Score"].idxmax()]
    print(f"\nMeilleure performance F1-Score:")
    print(f"  Durée: {best_f1['Duration']}s")
    print(f"  F1-Score: {best_f1['F1-Score']} ({best_f1['F1-Score_%']}%)")
    print(f"  Precision: {best_f1['Precision']}")
    print(f"  Recall: {best_f1['Recall']}")

    # Créer le dossier figures s'il n'existe pas
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)

    # Créer le graphique de performance
    plt.figure(figsize=(12, 8))

    # Subplot 1: F1-Score en pourcentage
    plt.subplot(2, 2, 1)
    plt.plot(df["Duration"], df["F1-Score_%"], "bo-", linewidth=2, markersize=8)
    plt.xlabel("Durée de fenêtre (secondes)")
    plt.ylabel("F1-Score (%)")
    plt.title("Performance F1-Score par Durée de Fenêtre")
    plt.grid(True, alpha=0.3)
    plt.xticks(df["Duration"])

    # Ajouter les valeurs sur le graphique
    for i, row in df.iterrows():
        plt.annotate(
            f"{row['F1-Score_%']}%",
            (row["Duration"], row["F1-Score_%"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # Subplot 2: Précision et Rappel
    plt.subplot(2, 2, 2)
    plt.plot(
        df["Duration"], df["Precision"] * 100, "ro-", label="Précision", linewidth=2
    )
    plt.plot(df["Duration"], df["Recall"] * 100, "go-", label="Rappel", linewidth=2)
    plt.xlabel("Durée de fenêtre (secondes)")
    plt.ylabel("Score (%)")
    plt.title("Précision et Rappel par Durée de Fenêtre")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(df["Duration"])

    # Subplot 3: ROC-AUC
    plt.subplot(2, 2, 3)
    plt.plot(df["Duration"], df["ROC-AUC"] * 100, "mo-", linewidth=2, markersize=8)
    plt.xlabel("Durée de fenêtre (secondes)")
    plt.ylabel("ROC-AUC (%)")
    plt.title("ROC-AUC par Durée de Fenêtre")
    plt.grid(True, alpha=0.3)
    plt.xticks(df["Duration"])

    # Subplot 4: Seuil optimal
    plt.subplot(2, 2, 4)
    plt.plot(df["Duration"], df["Optimal_Threshold"], "co-", linewidth=2, markersize=8)
    plt.xlabel("Durée de fenêtre (secondes)")
    plt.ylabel("Seuil Optimal")
    plt.title("Seuil Optimal par Durée de Fenêtre")
    plt.grid(True, alpha=0.3)
    plt.xticks(df["Duration"])

    plt.tight_layout()

    # Sauvegarder le graphique
    output_file = os.path.join(figures_dir, "performance_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nGraphique sauvegardé dans: {output_file}")

    # Créer aussi un graphique simple F1-Score uniquement
    plt.figure(figsize=(10, 6))
    plt.plot(df["Duration"], df["F1-Score_%"], "bo-", linewidth=3, markersize=10)
    plt.xlabel("Durée de fenêtre d'analyse (secondes)", fontsize=12)
    plt.ylabel("F1-Score (%)", fontsize=12)
    plt.title(
        "Performance F1-Score en fonction de la durée de fenêtre d'analyse", fontsize=14
    )
    plt.grid(True, alpha=0.3)
    plt.xticks(df["Duration"])

    # Ajouter les valeurs sur le graphique
    for i, row in df.iterrows():
        plt.annotate(
            f"{row['F1-Score_%']}%",
            (row["Duration"], row["F1-Score_%"]),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Marquer le maximum
    max_idx = df["F1-Score_%"].idxmax()
    max_duration = df.loc[max_idx, "Duration"]
    max_f1 = df.loc[max_idx, "F1-Score_%"]
    plt.scatter([max_duration], [max_f1], color="red", s=200, zorder=5)
    plt.annotate(
        f"Maximum: {max_f1}%\n({max_duration}s)",
        xy=(max_duration, max_f1),
        xytext=(max_duration + 5, max_f1 + 2),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        fontsize=12,
        color="red",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
    )

    plt.ylim(bottom=0)
    plt.tight_layout()

    # Sauvegarder le graphique simple
    simple_output_file = os.path.join(figures_dir, "f1_score_simple.png")
    plt.savefig(simple_output_file, dpi=300, bbox_inches="tight")
    print(f"Graphique F1-Score simple sauvegardé dans: {simple_output_file}")

    # Créer un tableau de résumé
    summary_file = os.path.join(figures_dir, "performance_summary.txt")
    with open(summary_file, "w") as f:
        f.write("RÉSUMÉ DE L'ANALYSE DE PERFORMANCE\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Nombre de durées testées: {len(df)}\n")
        f.write(
            f"Plage de durées: {df['Duration'].min()}s - {df['Duration'].max()}s\n\n"
        )
        f.write(f"MEILLEURE PERFORMANCE:\n")
        f.write(f"  Durée optimale: {best_f1['Duration']}s\n")
        f.write(f"  F1-Score: {best_f1['F1-Score']} ({best_f1['F1-Score_%']}%)\n")
        f.write(f"  Précision: {best_f1['Precision']}\n")
        f.write(f"  Rappel: {best_f1['Recall']}\n")
        f.write(f"  ROC-AUC: {best_f1['ROC-AUC']}\n\n")
        f.write("TABLEAU COMPLET:\n")
        f.write(df.to_string(index=False))

    print(f"Résumé textuel sauvegardé dans: {summary_file}")

else:
    print("Aucun résultat trouvé pour la comparaison")

print("\nFichiers générés:")
print("  - performance_comparison.csv (données tabulées)")
print("  - figures/performance_analysis.png (graphiques complets)")
print("  - figures/f1_score_simple.png (graphique F1-Score simple)")
print("  - figures/performance_summary.txt (résumé textuel)")
