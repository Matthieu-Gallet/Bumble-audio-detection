#!/usr/bin/env python3
"""
Script pour analyser les performances des données 2024 vs pollisophenocatch
Génère des figures comparatives du seuil optimal et du F1-score
"""

import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_metrics_txt(filepath):
    """Parse le fichier metrics.txt pour extraire le F1-score"""
    try:
        with open(filepath, "r") as f:
            content = f.read()

        # Chercher la ligne f1_score
        match = re.search(r"f1_score:\s*([\d.]+)", content)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"Erreur lecture {filepath}: {e}")
    return None


def parse_json_metrics(filepath):
    """Parse le fichier JSON pour extraire f1_score et optimal_threshold"""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        f1_score = data.get("f1_score", None)
        optimal_threshold = data.get("optimal_threshold", None)

        return f1_score, optimal_threshold
    except Exception as e:
        print(f"Erreur lecture {filepath}: {e}")
    return None, None


def analyze_output_final():
    """Analyse les données dans output_final_30"""
    base_dir = "output_final_30"

    # Dictionnaires pour stocker les résultats
    data_2024 = {}
    data_polli = {}

    # Parcourir tous les sous-dossiers
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        print(f"Traitement de {subdir}...")

        # Chemins vers les fichiers de métriques
        json_path = os.path.join(
            subdir_path,
            "advanced_evaluation",
            "advanced_Group_buzz",
            "advanced_metrics_Group_buzz.json",
        )
        txt_path = os.path.join(
            subdir_path, "classical_evaluation", "eval_Group_buzz", "metrics.txt"
        )

        # Vérifier que les fichiers existent
        if not os.path.exists(json_path):
            print(f"  Fichier JSON manquant: {json_path}")
            continue
        if not os.path.exists(txt_path):
            print(f"  Fichier TXT manquant: {txt_path}")
            continue

        # Extraire les métriques
        f1_classical = parse_metrics_txt(txt_path)
        f1_advanced, optimal_threshold = parse_json_metrics(json_path)

        if f1_classical is None or f1_advanced is None or optimal_threshold is None:
            print(f"  Données incomplètes pour {subdir}")
            continue

        # Stocker selon le type de données
        data_entry = {
            "f1_classical": f1_classical,
            "f1_advanced": f1_advanced,
            "optimal_threshold": optimal_threshold,
        }

        if subdir.startswith("2024"):
            data_2024[subdir] = data_entry
        elif subdir.startswith("polliso"):
            data_polli[subdir] = data_entry

        print(
            f"  F1 classical: {f1_classical:.3f}, F1 advanced: {f1_advanced:.3f}, Seuil: {optimal_threshold:.4f}"
        )

    return data_2024, data_polli


def create_comparison_plots(data_2024, data_polli):
    """Crée les figures de comparaison"""

    # Créer le dossier figures
    os.makedirs("figures", exist_ok=True)

    # Figure 1: Données 2024
    if data_2024:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(
            "Analyse des performances - Données 2024", fontsize=16, fontweight="bold"
        )

        # Préparer les données
        sessions = list(data_2024.keys())
        sessions_short = [
            s.replace("_session_", "_s").replace("_Tent_", "_").replace("SM0", "SM")
            for s in sessions
        ]
        f1_classical = [data_2024[s]["f1_classical"] for s in sessions]
        f1_advanced = [data_2024[s]["f1_advanced"] for s in sessions]
        thresholds = [data_2024[s]["optimal_threshold"] for s in sessions]

        x_pos = np.arange(len(sessions))

        # Subplot 1: F1-Scores
        ax1.plot(
            x_pos,
            np.array(f1_classical) * 100,
            "ro-",
            linewidth=2,
            markersize=8,
            label="F1 Classical (avant optimisation)",
            alpha=0.8,
        )
        ax1.plot(
            x_pos,
            np.array(f1_advanced) * 100,
            "bo-",
            linewidth=2,
            markersize=8,
            label="F1 Advanced (après optimisation)",
            alpha=0.8,
        )

        ax1.set_xlabel("Sessions 2024", fontsize=12)
        ax1.set_ylabel("F1-Score (%)", fontsize=12)
        ax1.set_title("Comparaison F1-Score: Classical vs Advanced", fontsize=14)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(sessions_short, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Ajouter les valeurs sur les points
        for i, (f1_c, f1_a) in enumerate(zip(f1_classical, f1_advanced)):
            ax1.annotate(
                f"{f1_c*100:.1f}%",
                (i, f1_c * 100),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color="red",
            )
            ax1.annotate(
                f"{f1_a*100:.1f}%",
                (i, f1_a * 100),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=9,
                color="blue",
            )

        # Subplot 2: Seuils optimaux
        ax2.plot(x_pos, thresholds, "go-", linewidth=2, markersize=8, alpha=0.8)
        ax2.set_xlabel("Sessions 2024", fontsize=12)
        ax2.set_ylabel("Seuil Optimal", fontsize=12)
        ax2.set_title("Seuil Optimal par Session", fontsize=14)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(sessions_short, rotation=45, ha="right")
        ax2.grid(True, alpha=0.3)

        # Ajouter les valeurs sur les points
        for i, thresh in enumerate(thresholds):
            ax2.annotate(
                f"{thresh:.3f}",
                (i, thresh),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig("figures/analysis_2024_sessions.png", dpi=300, bbox_inches="tight")
        print("Figure sauvegardée: figures/analysis_2024_sessions.png")
        plt.close()

    # Figure 2: Données Pollisophenocatch
    if data_polli:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(
            "Analyse des performances - Données Pollisophenocatch",
            fontsize=16,
            fontweight="bold",
        )

        # Préparer les données
        stations = list(data_polli.keys())
        stations_short = [
            s.replace("pollisophenocatch_SM0", "SM")
            .replace("_SM0", "_SM")
            .split("_")[1:3]
            for s in stations
        ]
        stations_short = [f"{parts[0]}_{parts[1]}" for parts in stations_short]

        f1_classical = [data_polli[s]["f1_classical"] for s in stations]
        f1_advanced = [data_polli[s]["f1_advanced"] for s in stations]
        thresholds = [data_polli[s]["optimal_threshold"] for s in stations]

        x_pos = np.arange(len(stations))

        # Subplot 1: F1-Scores
        ax1.plot(
            x_pos,
            np.array(f1_classical) * 100,
            "ro-",
            linewidth=2,
            markersize=8,
            label="F1 Classical (avant optimisation)",
            alpha=0.8,
        )
        ax1.plot(
            x_pos,
            np.array(f1_advanced) * 100,
            "bo-",
            linewidth=2,
            markersize=8,
            label="F1 Advanced (après optimisation)",
            alpha=0.8,
        )

        ax1.set_xlabel("Stations Pollisophenocatch", fontsize=12)
        ax1.set_ylabel("F1-Score (%)", fontsize=12)
        ax1.set_title("Comparaison F1-Score: Classical vs Advanced", fontsize=14)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(stations_short, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Ajouter les valeurs sur les points
        for i, (f1_c, f1_a) in enumerate(zip(f1_classical, f1_advanced)):
            ax1.annotate(
                f"{f1_c*100:.1f}%",
                (i, f1_c * 100),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color="red",
            )
            ax1.annotate(
                f"{f1_a*100:.1f}%",
                (i, f1_a * 100),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=9,
                color="blue",
            )

        # Subplot 2: Seuils optimaux
        ax2.plot(x_pos, thresholds, "go-", linewidth=2, markersize=8, alpha=0.8)
        ax2.set_xlabel("Stations Pollisophenocatch", fontsize=12)
        ax2.set_ylabel("Seuil Optimal", fontsize=12)
        ax2.set_title("Seuil Optimal par Station", fontsize=14)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(stations_short, rotation=45, ha="right")
        ax2.grid(True, alpha=0.3)

        # Ajouter les valeurs sur les points
        for i, thresh in enumerate(thresholds):
            ax2.annotate(
                f"{thresh:.3f}",
                (i, thresh),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(
            "figures/analysis_pollisophenocatch_stations.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Figure sauvegardée: figures/analysis_pollisophenocatch_stations.png")
        plt.close()


def create_summary_table(data_2024, data_polli):
    """Crée un tableau de résumé des résultats"""

    summary_file = "figures/performance_summary_comparison.txt"

    with open(summary_file, "w") as f:
        f.write("RÉSUMÉ DE L'ANALYSE COMPARATIVE\n")
        f.write("=" * 60 + "\n\n")

        # Données 2024
        if data_2024:
            f.write("DONNÉES 2024 (Sessions Terrain):\n")
            f.write("-" * 40 + "\n")
            for session, metrics in data_2024.items():
                f.write(f"{session}:\n")
                f.write(
                    f"  F1 Classical: {metrics['f1_classical']:.3f} ({metrics['f1_classical']*100:.1f}%)\n"
                )
                f.write(
                    f"  F1 Advanced:  {metrics['f1_advanced']:.3f} ({metrics['f1_advanced']*100:.1f}%)\n"
                )
                f.write(f"  Seuil Optimal: {metrics['optimal_threshold']:.4f}\n")
                improvement = (metrics["f1_advanced"] - metrics["f1_classical"]) * 100
                f.write(f"  Amélioration: {improvement:+.1f} points\n\n")

            # Statistiques générales 2024
            f1_classical_vals = [m["f1_classical"] for m in data_2024.values()]
            f1_advanced_vals = [m["f1_advanced"] for m in data_2024.values()]
            threshold_vals = [m["optimal_threshold"] for m in data_2024.values()]

            f.write("STATISTIQUES 2024:\n")
            f.write(
                f"  F1 Classical moyen: {np.mean(f1_classical_vals):.3f} ± {np.std(f1_classical_vals):.3f}\n"
            )
            f.write(
                f"  F1 Advanced moyen:  {np.mean(f1_advanced_vals):.3f} ± {np.std(f1_advanced_vals):.3f}\n"
            )
            f.write(
                f"  Seuil moyen: {np.mean(threshold_vals):.4f} ± {np.std(threshold_vals):.4f}\n"
            )
            f.write(
                f"  Amélioration moyenne: {(np.mean(f1_advanced_vals) - np.mean(f1_classical_vals))*100:+.1f} points\n\n"
            )

        # Données Pollisophenocatch
        if data_polli:
            f.write("DONNÉES POLLISOPHENOCATCH (Stations):\n")
            f.write("-" * 40 + "\n")
            for station, metrics in data_polli.items():
                station_short = station.replace("pollisophenocatch_", "").replace(
                    "_SM0", "_SM"
                )
                f.write(f"{station_short}:\n")
                f.write(
                    f"  F1 Classical: {metrics['f1_classical']:.3f} ({metrics['f1_classical']*100:.1f}%)\n"
                )
                f.write(
                    f"  F1 Advanced:  {metrics['f1_advanced']:.3f} ({metrics['f1_advanced']*100:.1f}%)\n"
                )
                f.write(f"  Seuil Optimal: {metrics['optimal_threshold']:.4f}\n")
                improvement = (metrics["f1_advanced"] - metrics["f1_classical"]) * 100
                f.write(f"  Amélioration: {improvement:+.1f} points\n\n")

            # Statistiques générales Pollisophenocatch
            f1_classical_vals = [m["f1_classical"] for m in data_polli.values()]
            f1_advanced_vals = [m["f1_advanced"] for m in data_polli.values()]
            threshold_vals = [m["optimal_threshold"] for m in data_polli.values()]

            f.write("STATISTIQUES POLLISOPHENOCATCH:\n")
            f.write(
                f"  F1 Classical moyen: {np.mean(f1_classical_vals):.3f} ± {np.std(f1_classical_vals):.3f}\n"
            )
            f.write(
                f"  F1 Advanced moyen:  {np.mean(f1_advanced_vals):.3f} ± {np.std(f1_advanced_vals):.3f}\n"
            )
            f.write(
                f"  Seuil moyen: {np.mean(threshold_vals):.4f} ± {np.std(threshold_vals):.4f}\n"
            )
            f.write(
                f"  Amélioration moyenne: {(np.mean(f1_advanced_vals) - np.mean(f1_classical_vals))*100:+.1f} points\n\n"
            )

    print(f"Résumé sauvegardé: {summary_file}")


def main():
    """Fonction principale"""
    print("Analyse des performances 2024 vs Pollisophenocatch")
    print("=" * 60)

    # Analyser les données
    data_2024, data_polli = analyze_output_final()

    print(f"\nDonnées trouvées:")
    print(f"  Sessions 2024: {len(data_2024)}")
    print(f"  Stations Pollisophenocatch: {len(data_polli)}")

    if not data_2024 and not data_polli:
        print("Aucune donnée trouvée!")
        return

    # Créer les figures
    print("\nGénération des figures...")
    create_comparison_plots(data_2024, data_polli)

    # Créer le résumé
    print("\nGénération du résumé...")
    create_summary_table(data_2024, data_polli)

    print("\nAnalyse terminée!")
    print("Fichiers générés:")
    print("  - figures/analysis_2024_sessions.png")
    print("  - figures/analysis_pollisophenocatch_stations.png")
    print("  - figures/performance_summary_comparison.txt")


if __name__ == "__main__":
    main()
