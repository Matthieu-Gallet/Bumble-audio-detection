#!/usr/bin/env python3
"""
Script d'analyse complète des résultats acoustiques
Analyse les groupes Buzz, Anthropophony et Geophony pour les 4 spots principaux
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time
import seaborn as sns
import warnings
import os
from pathlib import Path
import re

# Supprimer les warnings matplotlib
warnings.filterwarnings("ignore")

# Configuration des sites d'après select_acoustic_data.py
SITES_CONFIG = {
    "D01": {
        "serial": "SMA02939-D01",
        "location": "SERVOZ-DIOSAZ",
        "target": "Diosaz (riviere)",
        "altitude": 838,
        "color": "#1f77b4",
    },
    "D04": {
        "serial": "SMA02961-D04",
        "location": "PECLEREY-1400",
        "target": "Peclerey 1400 (Helico)",
        "altitude": 1400,
        "color": "#ff7f0e",
    },
    "D05": {
        "serial": "SMA02964-D05",
        "location": "LORIAZ-1630",
        "target": "Loriaz 1600 (voix)",
        "altitude": 1630,
        "color": "#2ca02c",
    },
    "D08": {
        "serial": "SMA02975-D08",
        "location": "LORIAZ-2140",
        "target": "Loriaz 2100 (vent)",
        "altitude": 2140,
        "color": "#d62728",
    },
}

# Colonnes d'analyse et seuils
ANALYSIS_GROUPS = {
    "Group_buzz": {
        "name": "Buzz (Insectes)",
        "thresholds": [0.3, 0.485],
        "color": "#8c564b",
    },
    "Group_anthropophony": {
        "name": "Anthropophony (Humain)",
        "thresholds": [0.3, 0.5],
        "color": "#e377c2",
    },
    "Group_geophony": {
        "name": "Geophony (Naturel)",
        "thresholds": [0.3, 0.5],
        "color": "#17becf",
    },
}

# Style matplotlib
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def load_and_prepare_data(csv_path):
    """
    Charge et prépare les données pour l'analyse
    """
    print("Chargement des donnees...")
    df = pd.read_csv(csv_path)

    # Convertir datetime (format: 20250410_102500)
    def parse_custom_datetime(dt_str):
        """Parse le format datetime personnalisé YYYYMMDD_HHMMSS"""
        try:
            return pd.to_datetime(dt_str, format="%Y%m%d_%H%M%S")
        except:
            # Fallback pour d'autres formats
            return pd.to_datetime(dt_str)

    df["datetime"] = df["datetime"].apply(parse_custom_datetime)

    # Extraire les informations temporelles
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["month_name"] = df["datetime"].dt.strftime("%B")

    # Identifier les spots à partir du nom de fichier
    def extract_spot_id(name):
        """Extrait l'ID du spot (D01, D04, D05, D08) depuis le nom"""
        for spot_id, config in SITES_CONFIG.items():
            if config["serial"] in name or spot_id in name:
                return spot_id
        return "Unknown"

    df["spot_id"] = df["name"].apply(extract_spot_id)

    # Filtrer les spots d'intérêt
    target_spots = list(SITES_CONFIG.keys())
    df = df[df["spot_id"].isin(target_spots)].copy()

    print(f"Donnees chargees: {len(df):,} enregistrements")
    print(f"   Spots: {df['spot_id'].value_counts().to_dict()}")
    print(f"   Période: {df['datetime'].min()} → {df['datetime'].max()}")

    return df


def plot_probability_time_series(df, output_dir):
    """
    Crée des graphiques de séries temporelles des probabilités pour chaque spot
    """
    print("\nCreation des series temporelles par spot...")

    os.makedirs(os.path.join(output_dir, "time_series"), exist_ok=True)

    for spot_id in SITES_CONFIG.keys():
        spot_data = df[df["spot_id"] == spot_id].copy()
        if len(spot_data) == 0:
            continue

        spot_config = SITES_CONFIG[spot_id]

        # Créer le graphique
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(
            f'Évolution des probabilités - {spot_config["target"]} ({spot_id})',
            fontsize=16,
            fontweight="bold",
        )

        for i, (group_col, group_info) in enumerate(ANALYSIS_GROUPS.items()):
            ax = axes[i]

            # Trier par date pour le graphique
            spot_data_sorted = spot_data.sort_values("datetime")

            # Graphique principal
            ax.plot(
                spot_data_sorted["datetime"],
                spot_data_sorted[group_col],
                alpha=0.7,
                linewidth=0.8,
                color=group_info["color"],
            )

            # Ajouter les lignes de seuil
            for threshold in group_info["thresholds"]:
                ax.axhline(
                    y=threshold,
                    color="red",
                    linestyle="--",
                    alpha=0.6,
                    label=f"Seuil {threshold}",
                )

            ax.set_title(f'{group_info["name"]}', fontweight="bold")
            ax.set_ylabel("Probabilité")
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Formater l'axe des dates
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        axes[-1].set_xlabel("Date")
        plt.tight_layout()

        # Sauvegarder
        output_path = os.path.join(
            output_dir, "time_series", f"probabilities_{spot_id}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"   {spot_config['target']}: {output_path}")


def calculate_hourly_detection_rates(df, group_col, threshold):
    """
    Calcule les taux de détection horaires par mois pour un groupe donné
    """
    # Créer les détections basées sur le seuil
    df_group = df.copy()
    df_group["detection"] = (df_group[group_col] >= threshold).astype(int)

    # Grouper par spot, mois et heure
    hourly_stats = (
        df_group.groupby(["spot_id", "month", "hour"])["detection"]
        .agg(
            [
                "mean",  # Taux de détection moyen
                "std",  # Écart-type
                "count",  # Nombre d'observations
            ]
        )
        .reset_index()
    )

    # Remplacer les NaN dans std par 0
    hourly_stats["std"] = hourly_stats["std"].fillna(0)

    return hourly_stats


def plot_monthly_hourly_detection_rates(df, output_dir):
    """
    Crée les graphiques de taux de détection horaires par mois
    """
    print("\nCreation des graphiques de taux de detection...")

    # Définir les mois
    months = {4: "Avril", 5: "Mai", 6: "Juin"}

    for group_col, group_info in ANALYSIS_GROUPS.items():
        print(f"\n   Analyse de {group_info['name']}...")

        os.makedirs(
            os.path.join(output_dir, "detection_rates", group_col), exist_ok=True
        )

        for threshold in group_info["thresholds"]:
            # Calculer les statistiques
            hourly_stats = calculate_hourly_detection_rates(df, group_col, threshold)

            # Calculer le maximum des valeurs moyennes pour ajuster l'axe Y
            max_mean_value = 0
            all_month_data = {}
            for month_num in months.keys():
                month_data = hourly_stats[hourly_stats["month"] == month_num]
                all_month_data[month_num] = month_data
                if len(month_data) > 0:
                    max_mean_value = max(max_mean_value, month_data["mean"].max())

            # Ajouter une marge de 10% au maximum
            y_max = max_mean_value * 1.1 if max_mean_value > 0 else 0.1

            # Créer un graphique pour chaque mois avec sharex et sharey
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
            fig.suptitle(
                f'{group_info["name"]} - Taux de détection horaires (seuil: {threshold})',
                fontsize=16,
                fontweight="bold",
            )

            for i, (month_num, month_name) in enumerate(months.items()):
                ax = axes[i]

                # Données du mois
                month_data = all_month_data[month_num]

                if len(month_data) == 0:
                    ax.text(
                        0.5,
                        0.5,
                        f"Pas de données\npour {month_name}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=12,
                    )
                    ax.set_title(f"{month_name} 2025")
                    continue

                # Graphique pour chaque spot
                for spot_id in SITES_CONFIG.keys():
                    spot_data = month_data[month_data["spot_id"] == spot_id]
                    if len(spot_data) == 0:
                        continue

                    spot_config = SITES_CONFIG[spot_id]

                    # Créer une série complète d'heures (8h-21h)
                    hours = range(8, 22)  # 8h à 21h
                    means = []
                    stds = []

                    for hour in hours:
                        hour_data = spot_data[spot_data["hour"] == hour]
                        if len(hour_data) > 0:
                            means.append(hour_data["mean"].iloc[0])
                            stds.append(hour_data["std"].iloc[0])
                        else:
                            means.append(0)
                            stds.append(0)

                    means = np.array(means)
                    stds = np.array(stds)

                    # Graphique principal
                    ax.plot(
                        hours,
                        means,
                        "o-",
                        label=f"{spot_id} ({spot_config['target'].split('(')[0].strip()})",
                        color=spot_config["color"],
                        linewidth=2,
                        markersize=6,
                    )

                    # Bande de confiance (moyenne ± écart-type)
                    ax.fill_between(
                        hours,
                        np.maximum(0, means - stds),
                        np.minimum(y_max, means + stds),
                        alpha=0.2,
                        color=spot_config["color"],
                    )

                ax.set_title(f"{month_name} 2025", fontweight="bold")
                ax.set_xlabel("Heure de la journée")
                if i == 0:  # Seulement sur le premier graphique
                    ax.set_ylabel("Taux de détection")
                ax.set_xlim(7.5, 21.5)
                ax.set_ylim(0, y_max)
                ax.grid(True, alpha=0.3)

                # Légende seulement sur le dernier graphique (juin)
                if i == 2:  # Dernier graphique (juin)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

                # Format des heures
                ax.set_xticks(range(8, 22, 2))
                ax.set_xticklabels([f"{h}h" for h in range(8, 22, 2)])

            plt.tight_layout()

            # Sauvegarder
            threshold_str = str(threshold).replace(".", "_")
            output_path = os.path.join(
                output_dir,
                "detection_rates",
                group_col,
                f"monthly_hourly_{group_col}_threshold_{threshold_str}.png",
            )
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"      Seuil {threshold}: {output_path}")


def create_summary_statistics(df, output_dir):
    """
    Crée un rapport de statistiques résumées
    """
    print("\nGeneration des statistiques resumees...")

    summary_stats = []

    for group_col, group_info in ANALYSIS_GROUPS.items():
        for threshold in group_info["thresholds"]:
            for spot_id in SITES_CONFIG.keys():
                spot_data = df[df["spot_id"] == spot_id]
                if len(spot_data) == 0:
                    continue

                spot_config = SITES_CONFIG[spot_id]
                detections = (spot_data[group_col] >= threshold).astype(int)

                # Statistiques par mois
                for month in [4, 5, 6]:
                    month_data = spot_data[spot_data["month"] == month]
                    if len(month_data) == 0:
                        continue

                    month_detections = (month_data[group_col] >= threshold).astype(int)

                    summary_stats.append(
                        {
                            "Groupe": group_info["name"],
                            "Seuil": threshold,
                            "Spot": spot_id,
                            "Site": spot_config["target"],
                            "Mois": ["Avril", "Mai", "Juin"][month - 4],
                            "Nb_Enregistrements": len(month_data),
                            "Nb_Détections": month_detections.sum(),
                            "Taux_Détection": month_detections.mean(),
                            "Proba_Moyenne": month_data[group_col].mean(),
                            "Proba_Médiane": month_data[group_col].median(),
                            "Proba_Max": month_data[group_col].max(),
                        }
                    )

    # Créer le DataFrame des statistiques
    stats_df = pd.DataFrame(summary_stats)

    # Sauvegarder en CSV
    stats_path = os.path.join(output_dir, "summary_statistics.csv")
    stats_df.to_csv(stats_path, index=False)

    print(f"   Statistiques sauvees: {stats_path}")

    return stats_df


def main():
    """
    Fonction principale d'analyse
    """
    # Chemins
    csv_path = "/mnt/BACK UP/inference/merged_results.csv"
    output_dir = "acoustic_analysis_results"

    print("ANALYSE COMPLETE DES RESULTATS ACOUSTIQUES")
    print("=" * 80)
    print(f"Fichier source: {csv_path}")
    print(f"Répertoire de sortie: {output_dir}")

    # Vérifier que le fichier existe
    if not os.path.exists(csv_path):
        print(f"❌ ERREUR: Fichier non trouvé: {csv_path}")
        return

    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)

    # Charger les données
    df = load_and_prepare_data(csv_path)

    # Analyses
    plot_probability_time_series(df, output_dir)
    plot_monthly_hourly_detection_rates(df, output_dir)
    stats_df = create_summary_statistics(df, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSE TERMINEE!")
    print(f"Resultats disponibles dans: {os.path.abspath(output_dir)}")
    print("\nResume des analyses generees:")
    print("   • Séries temporelles des probabilités par spot")
    print("   • Taux de détection horaires par mois (3 groupes × 2 seuils)")
    print("   • Statistiques résumées (CSV)")
    print("   • Heatmaps de comparaison par site et mois")
    print(f"\nDonnees analysees: {len(df):,} enregistrements")
    print(f"Sites: {', '.join(SITES_CONFIG.keys())}")
    print(
        f"Periode: {df['datetime'].min().strftime('%d/%m/%Y')} → {df['datetime'].max().strftime('%d/%m/%Y')}"
    )


if __name__ == "__main__":
    main()
