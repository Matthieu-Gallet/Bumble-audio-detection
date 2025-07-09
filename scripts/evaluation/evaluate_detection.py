#!/usr/bin/env python3
"""
Script pour comparer les résultats de détection automatique avec la vérité terrain
et calculer les métriques de performance en machine learning.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import seaborn as sns
from typing import List, Tuple, Dict


def load_ground_truth(annotations_dir: str) -> pd.DataFrame:
    """
    Charger les annotations de vérité terrain depuis les fichiers texte.

    Args:
        annotations_dir: Chemin vers le dossier contenant les fichiers d'annotations

    Returns:
        DataFrame avec les colonnes: filename, start_time, end_time, label
    """
    ground_truth = []

    for txt_file in Path(annotations_dir).glob("*.txt"):
        filename = txt_file.stem + ".wav"  # Nom du fichier audio correspondant

        with open(txt_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        start_time = float(parts[0])
                        end_time = float(parts[1])
                        # Extraire le label depuis le nom du fichier (ex: C048_F_T pour buzz)
                        label = parts[2] if len(parts) > 2 else "buzz"

                        ground_truth.append(
                            {
                                "filename": filename,
                                "start_time": start_time,
                                "end_time": end_time,
                                "label": "buzz",  # Assumons que toutes les annotations sont des buzz
                            }
                        )

    return pd.DataFrame(ground_truth)


def load_predictions(
    csv_path: str,
    detection_column: str = "tag_Buzz",
    threshold: float = 0.5,
    duration: float = 10.0,
) -> pd.DataFrame:
    """
    Charger les prédictions depuis le fichier CSV de résultats.

    Args:
        csv_path: Chemin vers le fichier CSV des résultats
        detection_column: Nom de la colonne contenant les scores de détection
        threshold: Seuil pour considérer une détection comme positive

    Returns:
        DataFrame avec les prédictions
    """
    df = pd.read_csv(csv_path)

    # Convertir les scores en prédictions binaires
    df["prediction"] = (df[detection_column] > threshold).astype(int)
    df["score"] = df[detection_column]

    # Calculer les temps de fin basés sur la durée des segments (10 secondes par défaut)
    df["end_time"] = df["start"] + duration

    return df[["name", "start", "end_time", "prediction", "score"]]


def create_time_segments(
    ground_truth: pd.DataFrame, predictions: pd.DataFrame, segment_length: float = 10.0
) -> pd.DataFrame:
    """
    Créer des segments de temps alignés pour la comparaison.

    Args:
        ground_truth: DataFrame avec les annotations
        predictions: DataFrame avec les prédictions
        segment_length: Longueur des segments en secondes

    Returns:
        DataFrame avec les segments alignés
    """
    results = []

    # Obtenir tous les fichiers uniques
    all_files = set(ground_truth["filename"].unique()) | set(
        predictions["name"].unique()
    )

    for filename in all_files:
        # Filtrer les données pour ce fichier
        gt_file = ground_truth[ground_truth["filename"] == filename]
        pred_file = predictions[predictions["name"] == filename]

        # Déterminer la durée totale du fichier
        max_time = 0
        if not gt_file.empty:
            max_time = max(max_time, gt_file["end_time"].max())
        if not pred_file.empty:
            max_time = max(max_time, pred_file["end_time"].max())

        # Créer des segments de temps
        for start_time in np.arange(0, max_time, segment_length):
            end_time = start_time + segment_length

            # Vérifier si il y a une annotation dans ce segment
            gt_overlap = gt_file[
                (gt_file["start_time"] < end_time) & (gt_file["end_time"] > start_time)
            ]
            ground_truth_label = 1 if len(gt_overlap) > 0 else 0

            # Obtenir la prédiction pour ce segment
            pred_segment = pred_file[
                (pred_file["start"] <= start_time)
                & (pred_file["end_time"] > start_time)
            ]

            if len(pred_segment) > 0:
                prediction = pred_segment.iloc[0]["prediction"]
                score = pred_segment.iloc[0]["score"]
            else:
                prediction = 0
                score = 0.0

            results.append(
                {
                    "filename": filename,
                    "start_time": start_time,
                    "end_time": end_time,
                    "ground_truth": ground_truth_label,
                    "prediction": prediction,
                    "score": score,
                }
            )

    return pd.DataFrame(results)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculer les métriques de performance.

    Args:
        y_true: Vérité terrain
        y_pred: Prédictions

    Returns:
        Dictionnaire avec les métriques
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculs supplémentaires
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "support": support,
    }


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None
):
    """
    Tracer la matrice de confusion.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Buzz", "Buzz"],
        yticklabels=["No Buzz", "Buzz"],
    )
    plt.title("Matrice de Confusion")
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité Terrain")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.show()


def plot_precision_recall_curve(segments_df: pd.DataFrame, save_path: str = None):
    """
    Tracer la courbe précision-rappel en variant le seuil.
    """
    thresholds = np.linspace(0, 1, 101)
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        y_pred = (segments_df["score"] > threshold).astype(int)
        metrics = calculate_metrics(segments_df["ground_truth"], y_pred)
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1_scores.append(metrics["f1_score"])

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(thresholds, precisions, "b-", label="Précision")
    plt.plot(thresholds, recalls, "r-", label="Rappel")
    plt.xlabel("Seuil")
    plt.ylabel("Score")
    plt.title("Précision et Rappel vs Seuil")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(recalls, precisions, "g-")
    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.title("Courbe Précision-Rappel")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(thresholds, f1_scores, "m-")
    plt.xlabel("Seuil")
    plt.ylabel("F1-Score")
    plt.title("F1-Score vs Seuil")
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.show()


def analyze_detection_performance(
    csv_path: str,
    annotations_dir: str,
    detection_column: str = "tag_Buzz",
    threshold: float = 0.5,
    output_dir: str = None,
    duration: float = 10.0,
):
    """
    Fonction principale pour analyser les performances de détection.

    Args:
        csv_path: Chemin vers le fichier CSV des résultats
        annotations_dir: Chemin vers le dossier d'annotations
        detection_column: Colonne contenant les scores de détection
        threshold: Seuil pour la détection
        output_dir: Dossier pour sauvegarder les résultats
    """
    print("Loading data...")

    # Charger les données
    ground_truth = load_ground_truth(annotations_dir)
    predictions = load_predictions(
        csv_path, detection_column, threshold, duration=duration
    )

    print(f"Ground truth: {len(ground_truth)} annotations")
    print(f"Prédictions: {len(predictions)} segments")

    # Créer les segments alignés
    segments_df = create_time_segments(ground_truth, predictions)

    print(f"⏱️ Segments créés: {len(segments_df)}")
    print(f"Positive segments (ground truth): {segments_df['ground_truth'].sum()}")
    print(f"Detected segments: {segments_df['prediction'].sum()}")

    # Calculer les métriques
    metrics = calculate_metrics(segments_df["ground_truth"], segments_df["prediction"])

    print("\n📈 MÉTRIQUES DE PERFORMANCE:")
    print("=" * 50)
    print(f"Précision:        {metrics['precision']:.3f}")
    print(f"Rappel:           {metrics['recall']:.3f}")
    print(f"F1-Score:         {metrics['f1_score']:.3f}")
    print(f"Accuracy:         {metrics['accuracy']:.3f}")
    print(f"Spécificité:      {metrics['specificity']:.3f}")
    print(f"Vrais Positifs:   {metrics['true_positives']}")
    print(f"Faux Positifs:    {metrics['false_positives']}")
    print(f"Vrais Négatifs:   {metrics['true_negatives']}")
    print(f"Faux Négatifs:    {metrics['false_negatives']}")

    # Créer les graphiques
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        confusion_path = os.path.join(output_dir, "confusion_matrix.png")
        pr_curve_path = os.path.join(output_dir, "precision_recall_curve.png")
    else:
        confusion_path = None
        pr_curve_path = None

    plot_confusion_matrix(
        segments_df["ground_truth"], segments_df["prediction"], confusion_path
    )
    plot_precision_recall_curve(segments_df, pr_curve_path)

    # Sauvegarder les résultats détaillés
    if output_dir:
        results_path = os.path.join(output_dir, "detailed_results.csv")
        segments_df.to_csv(results_path, index=False)
        print(f"\n💾 Résultats détaillés sauvegardés: {results_path}")

        # Sauvegarder les métriques
        metrics_path = os.path.join(output_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("MÉTRIQUES DE PERFORMANCE\n")
            f.write("=" * 50 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        print(f"Metrics saved: {metrics_path}")

    return segments_df, metrics


if __name__ == "__main__":
    # Configuration using relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))

    CSV_PATH = os.path.join(parent_dir, "output", "indices_test.csv")
    ANNOTATIONS_DIR = os.path.join(
        parent_dir, "data", "20240408_session_01_Tent", "SM05_T_annotées"
    )
    DETECTION_COLUMN = "tag_Buzz"  # Column for buzz detection
    THRESHOLD = 0.5  # Detection threshold
    OUTPUT_DIR = os.path.join(parent_dir, "output", "evaluation")

    # Launch analysis
    segments_df, metrics = analyze_detection_performance(
        CSV_PATH,
        ANNOTATIONS_DIR,
        DETECTION_COLUMN,
        THRESHOLD,
        OUTPUT_DIR,
        duration=10.0,
    )

    print("\nAnalysis completed!")
