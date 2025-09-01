#!/usr/bin/env python3
"""
Script avancé pour évaluer les performances de détection avec optimisation du seuil,
métriques complètes, et visualisations avancées.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    precision_recall_fscore_support,
    average_precision_score,
)
from pathlib import Path
import json
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


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


def optimize_threshold(
    y_true: np.ndarray, y_scores: np.ndarray, metric: str = "f1"
) -> Dict:
    """
    Optimiser le seuil de détection pour maximiser une métrique donnée.

    Args:
        y_true: Vérité terrain
        y_scores: Scores de prédiction (probabilités)
        metric: Métrique à optimiser ('f1', 'precision', 'recall')

    Returns:
        Dictionnaire avec le seuil optimal et les métriques associées
    """
    thresholds = np.linspace(0.001, 1.0, 200)
    best_threshold = 0.5
    best_score = 0
    threshold_results = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        if len(np.unique(y_pred)) == 1:  # Tous les mêmes prédictions
            continue

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        threshold_results.append(
            {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1}
        )

        if metric == "f1" and f1 > best_score:
            best_score = f1
            best_threshold = threshold
        elif metric == "precision" and precision > best_score:
            best_score = precision
            best_threshold = threshold
        elif metric == "recall" and recall > best_score:
            best_score = recall
            best_threshold = threshold

    return {
        "optimal_threshold": best_threshold,
        "best_score": best_score,
        "metric_optimized": metric,
        "threshold_results": pd.DataFrame(threshold_results),
    }


def calculate_advanced_metrics(
    y_true: np.ndarray, y_scores: np.ndarray, y_pred: np.ndarray
) -> Dict:
    """
    Calculer des métriques avancées incluant ROC-AUC, PR-AUC, weighted F1, etc.

    Args:
        y_true: Vérité terrain
        y_scores: Scores de prédiction
        y_pred: Prédictions binaires

    Returns:
        Dictionnaire avec les métriques avancées
    """
    metrics = {}

    # Métriques de base
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # Weighted metrics (utile si les classes sont déséquilibrées)
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )

    # Macro average
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculs supplémentaires
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # = recall

    # ROC-AUC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
    except:
        roc_auc = 0.5

    # Precision-Recall AUC
    try:
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(pr_recall, pr_precision)
        average_precision = average_precision_score(y_true, y_scores)
    except:
        pr_auc = 0.0
        average_precision = 0.0

    return {
        # Métriques de base
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "sensitivity": sensitivity,
        # Métriques weighted (pour classes déséquilibrées)
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        # Métriques macro
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        # AUC metrics
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "average_precision": average_precision,
        # Confusion matrix
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "support": support,
        # Données pour plots
        "fpr": fpr if "fpr" in locals() else None,
        "tpr": tpr if "tpr" in locals() else None,
        "pr_precision": pr_precision if "pr_precision" in locals() else None,
        "pr_recall": pr_recall if "pr_recall" in locals() else None,
    }


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

    print(f"Segments créés: {len(segments_df)}")

    # Calculate metrics for each column
    print(f"Positive segments (ground truth): {segments_df['ground_truth'].sum()}")
    print(f"Detected segments: {segments_df['prediction'].sum()}")

    # Calculer les métriques
    metrics = calculate_metrics(segments_df["ground_truth"], segments_df["prediction"])

    print("\nMÉTRIQUES DE PERFORMANCE:")
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
        print(f"\nRésultats détaillés sauvegardés: {results_path}")

        # Sauvegarder les métriques
        metrics_path = os.path.join(output_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("MÉTRIQUES DE PERFORMANCE\n")
            f.write("=" * 50 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        print(f"Metrics saved: {metrics_path}")

    return segments_df, metrics


def create_error_analysis_by_class(
    csv_path,
    segments_df,
    ground_truth,
    final_predictions,
    output_dir,
    column_name,
    excluded_classes=None,
):
    """
    Create visualization of false positives and false negatives by class.

    Args:
        csv_path: Path to original CSV file with all prediction columns
        segments_df: DataFrame with segments (for indexing)
        ground_truth: Array of ground truth labels (0/1)
        final_predictions: Array of final binary predictions (0/1)
        output_dir: Directory to save plots
        column_name: Name of the column being evaluated
        excluded_classes: Dict with classes to exclude from analysis for each target class
    """
    # Load the original CSV to get all prediction columns
    try:
        original_df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Could not load original CSV for error analysis: {e}")
        return

    # Define all prediction columns available
    prediction_columns = [
        "tag_fly_housefly",
        "tag_bee_wasp",
        "tag_fly_bee_wasp",
        "tag_water",
        "tag_wind",
        "tag_motor_vehicle",
        "tag_aircraft",
        "tag_human_voice",
        "Group_buzz",
        "Group_geophony",
        "Group_anthropophony",
    ]

    # Filter columns that actually exist in the dataframe
    available_columns = [
        col for col in prediction_columns if col in original_df.columns
    ]

    # Remove excluded classes for this target column if specified
    if excluded_classes and column_name in excluded_classes:
        excluded_for_column = excluded_classes[column_name]
        available_columns = [
            col for col in available_columns if col not in excluded_for_column
        ]
        print(f"Excluded classes for {column_name}: {excluded_for_column}")

    if not available_columns:
        print("No prediction columns found for error analysis")
        return

    # Create a mapping from segments to original data
    # This is tricky because segments might not align perfectly with original rows
    # For now, we'll use the filename and start time to match
    if "filename" not in segments_df.columns or "start_time" not in segments_df.columns:
        print("Missing filename or start_time columns for error analysis mapping")
        return

    # Identify false positives and false negatives
    false_positives = (ground_truth == 0) & (final_predictions == 1)
    false_negatives = (ground_truth == 1) & (final_predictions == 0)

    print(f"\nANALYSE DES ERREURS PAR CLASSE")
    print("-" * 50)
    print(f"Faux positifs: {false_positives.sum()}")
    print(f"Faux négatifs: {false_negatives.sum()}")

    if false_positives.sum() == 0 and false_negatives.sum() == 0:
        print("Aucune erreur détectée !")
        return

    # Create figure for error analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Analyse des erreurs par classe - {column_name}", fontsize=16)

    # Function to get prediction data for error segments
    def get_error_predictions(error_mask, error_type):
        error_segments = segments_df[error_mask]
        error_predictions = []

        for _, segment in error_segments.iterrows():
            # Find matching rows in original data
            filename = segment["filename"]
            start_time = segment["start_time"]

            # Look for matching rows in original CSV
            matching_rows = original_df[
                (
                    original_df["name"].str.contains(
                        filename.replace(".wav", ""), na=False
                    )
                )
                & (abs(original_df["start"] - start_time) < 2.5)  # Allow 2.5s tolerance
            ]

            if not matching_rows.empty:
                # Use the first matching row
                row = matching_rows.iloc[0]

                # Get the threshold for this row
                used_threshold = row.get(
                    "used_threshold", 0.5
                )  # Default to 0.5 if column doesn't exist

                # Find classes that exceed the threshold
                classes_above_threshold = []
                for col in available_columns:
                    if row[col] > used_threshold:
                        classes_above_threshold.append((col, row[col]))

                # If no class exceeds threshold, assign 'none'
                if not classes_above_threshold:
                    error_predictions.append(("none", 0.0))
                else:
                    # Take the first class that exceeds threshold (or could be max if multiple)
                    # For consistency with the provided example, take the first one found
                    predicted_class, score = classes_above_threshold[0]
                    error_predictions.append((predicted_class, score))

        return error_predictions

    # 1. False Positives - What classes are detected instead?
    if false_positives.sum() > 0:
        fp_predictions = get_error_predictions(false_positives, "False Positives")

        if fp_predictions:
            fp_classes = [pred[0] for pred in fp_predictions]
            fp_class_counts = pd.Series(fp_classes).value_counts()
            fp_percentages = (fp_class_counts / fp_class_counts.sum() * 100).round(1)

            # Plot false positives by class
            axes[0, 0].bar(
                range(len(fp_percentages)),
                fp_percentages.values,
                color="red",
                alpha=0.7,
            )
            axes[0, 0].set_title(
                f"Faux Positifs par Classe\n(Total: {false_positives.sum()})"
            )
            axes[0, 0].set_ylabel("Pourcentage (%)")
            axes[0, 0].set_xticks(range(len(fp_percentages)))
            axes[0, 0].set_xticklabels(fp_percentages.index, rotation=45, ha="right")
            axes[0, 0].grid(True, alpha=0.3)

            # Add percentage labels on bars
            for i, v in enumerate(fp_percentages.values):
                axes[0, 0].text(i, v + 0.5, f"{v}%", ha="center", va="bottom")

            # Print detailed false positives info
            print(f"\nFAUX POSITIFS (classes détectées à tort):")
            for class_name, percentage in fp_percentages.items():
                count = fp_class_counts[class_name]
                print(f"  {class_name}: {count} ({percentage}%)")
        else:
            axes[0, 0].text(
                0.5,
                0.5,
                "Données non disponibles",
                ha="center",
                va="center",
                transform=axes[0, 0].transAxes,
                fontsize=14,
            )
    else:
        axes[0, 0].text(
            0.5,
            0.5,
            "Aucun faux positif",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
            fontsize=14,
        )

    axes[0, 0].set_title("Faux Positifs par Classe")

    # 2. False Negatives - What classes are detected instead?
    if false_negatives.sum() > 0:
        fn_predictions = get_error_predictions(false_negatives, "False Negatives")

        if fn_predictions:
            fn_classes = [pred[0] for pred in fn_predictions]
            fn_class_counts = pd.Series(fn_classes).value_counts()
            fn_percentages = (fn_class_counts / fn_class_counts.sum() * 100).round(1)

            # Plot false negatives by class
            axes[0, 1].bar(
                range(len(fn_percentages)),
                fn_percentages.values,
                color="orange",
                alpha=0.7,
            )
            axes[0, 1].set_title(
                f"Faux Négatifs par Classe\n(Total: {false_negatives.sum()})"
            )
            axes[0, 1].set_ylabel("Pourcentage (%)")
            axes[0, 1].set_xticks(range(len(fn_percentages)))
            axes[0, 1].set_xticklabels(fn_percentages.index, rotation=45, ha="right")
            axes[0, 1].grid(True, alpha=0.3)

            # Add percentage labels on bars
            for i, v in enumerate(fn_percentages.values):
                axes[0, 1].text(i, v + 0.5, f"{v}%", ha="center", va="bottom")

            # Print detailed false negatives info
            print(f"\nFAUX NÉGATIFS (classes détectées à la place):")
            for class_name, percentage in fn_percentages.items():
                count = fn_class_counts[class_name]
                print(f"  {class_name}: {count} ({percentage}%)")
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "Données non disponibles",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
                fontsize=14,
            )
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "Aucun faux négatif",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
            fontsize=14,
        )

    axes[0, 1].set_title("Faux Négatifs par Classe")

    # 3. Distribution of error scores for the target column
    if column_name in original_df.columns:
        fp_segments = segments_df[false_positives]
        tp_mask = (ground_truth == 1) & (final_predictions == 1)
        tp_segments = segments_df[tp_mask]

        fp_scores = []
        tp_scores = []

        # Get scores for false positives
        for _, segment in fp_segments.iterrows():
            filename = segment["filename"]
            start_time = segment["start_time"]
            matching_rows = original_df[
                (
                    original_df["name"].str.contains(
                        filename.replace(".wav", ""), na=False
                    )
                )
                & (abs(original_df["start"] - start_time) < 2.5)
            ]
            if not matching_rows.empty:
                fp_scores.append(matching_rows.iloc[0][column_name])

        # Get scores for true positives
        for _, segment in tp_segments.iterrows():
            filename = segment["filename"]
            start_time = segment["start_time"]
            matching_rows = original_df[
                (
                    original_df["name"].str.contains(
                        filename.replace(".wav", ""), na=False
                    )
                )
                & (abs(original_df["start"] - start_time) < 2.5)
            ]
            if not matching_rows.empty:
                tp_scores.append(matching_rows.iloc[0][column_name])

        if fp_scores and tp_scores:
            axes[1, 0].hist(
                fp_scores,
                bins=30,
                alpha=0.7,
                label="Faux Positifs",
                color="red",
                density=True,
            )
            axes[1, 0].hist(
                tp_scores,
                bins=30,
                alpha=0.7,
                label="Vrais Positifs",
                color="green",
                density=True,
            )
            axes[1, 0].set_xlabel("Score de Prédiction")
            axes[1, 0].set_ylabel("Densité")
            axes[1, 0].set_title(f"Distribution des Scores\n({column_name})")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "Données non disponibles",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
                fontsize=14,
            )
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            f"Colonne {column_name} non trouvée",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
            fontsize=14,
        )

    # 4. Summary text
    axes[1, 1].text(
        0.1,
        0.8,
        f"Analyse des erreurs pour {column_name}:",
        transform=axes[1, 1].transAxes,
        fontsize=12,
        weight="bold",
    )
    axes[1, 1].text(
        0.1,
        0.6,
        f"• Faux positifs: {false_positives.sum()}",
        transform=axes[1, 1].transAxes,
        fontsize=11,
    )
    axes[1, 1].text(
        0.1,
        0.4,
        f"• Faux négatifs: {false_negatives.sum()}",
        transform=axes[1, 1].transAxes,
        fontsize=11,
    )
    axes[1, 1].text(
        0.1,
        0.2,
        f"• Classes analysées: {len(available_columns)}",
        transform=axes[1, 1].transAxes,
        fontsize=11,
    )

    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Résumé")

    plt.tight_layout()

    # Save error analysis plot
    error_plot_path = os.path.join(output_dir, f"error_analysis_{column_name}.png")
    plt.savefig(error_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Analyse des erreurs sauvegardée: {error_plot_path}")


def create_advanced_plots(ground_truth, predictions, scores, output_dir, column_name):
    """
    Create advanced visualizations including ROC curve, precision-recall curve,
    threshold analysis, and distribution plots.

    Args:
        ground_truth: Array of ground truth labels (0/1)
        predictions: Array of binary predictions (0/1)
        scores: Array of prediction scores/probabilities
        output_dir: Directory to save plots
        column_name: Name of the column being evaluated
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Advanced Analysis for {column_name}", fontsize=16)

    # 1. ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(ground_truth, scores)
    roc_auc = auc(fpr, tpr)

    axes[0, 0].plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    axes[0, 0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(ground_truth, scores)
    pr_auc = auc(recall, precision)

    axes[0, 1].plot(
        recall, precision, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})"
    )
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("Precision-Recall Curve")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Score Distribution
    positive_scores = scores[ground_truth == 1]
    negative_scores = scores[ground_truth == 0]

    axes[0, 2].hist(
        negative_scores, bins=50, alpha=0.7, label="Negative", color="red", density=True
    )
    axes[0, 2].hist(
        positive_scores,
        bins=50,
        alpha=0.7,
        label="Positive",
        color="green",
        density=True,
    )
    axes[0, 2].set_xlabel("Prediction Score")
    axes[0, 2].set_ylabel("Density")
    axes[0, 2].set_title("Score Distribution")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. F1-Score vs Threshold
    f1_scores = []
    thresholds_range = np.linspace(0, 1, 100)

    for thresh in thresholds_range:
        thresh_predictions = (scores >= thresh).astype(int)
        f1 = f1_score(ground_truth, thresh_predictions, zero_division=0)
        f1_scores.append(f1)

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds_range[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]

    axes[1, 0].plot(thresholds_range, f1_scores, "b-", linewidth=2)
    axes[1, 0].axvline(
        x=optimal_threshold,
        color="red",
        linestyle="--",
        label=f"Optimal: {optimal_threshold:.3f} (F1={optimal_f1:.3f})",
    )
    axes[1, 0].set_xlabel("Threshold")
    axes[1, 0].set_ylabel("F1-Score")
    axes[1, 0].set_title("F1-Score vs Threshold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Confusion Matrix (current threshold)
    cm = confusion_matrix(ground_truth, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 1])
    axes[1, 1].set_title("Confusion Matrix")
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("Actual")

    # 6. Metrics Summary
    metrics = calculate_advanced_metrics(ground_truth, scores, predictions)

    metrics_text = f"""
    Precision: {metrics['precision']:.3f}
    Recall: {metrics['recall']:.3f}
    F1-Score: {metrics['f1_score']:.3f}
    Weighted F1: {metrics['weighted_f1']:.3f}
    Accuracy: {metrics['accuracy']:.3f}
    Specificity: {metrics['specificity']:.3f}
    ROC AUC: {roc_auc:.3f}
    PR AUC: {pr_auc:.3f}
    
    Optimal Threshold: {optimal_threshold:.3f}
    Optimal F1: {optimal_f1:.3f}
    """

    axes[1, 2].text(
        0.1,
        0.5,
        metrics_text,
        transform=axes[1, 2].transAxes,
        fontsize=12,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis("off")
    axes[1, 2].set_title("Metrics Summary")

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, f"advanced_analysis_{column_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Advanced plots saved: {plot_path}")

    return {
        "optimal_threshold": optimal_threshold,
        "optimal_f1": optimal_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def optimize_threshold_by_f1(ground_truth, scores, num_points=100):
    """
    Find optimal threshold that maximizes F1-Score.

    Args:
        ground_truth: Array of ground truth labels (0/1)
        scores: Array of prediction scores/probabilities
        num_points: Number of threshold points to test

    Returns:
        dict: Contains optimal_threshold, optimal_f1, and all_thresholds/f1_scores
    """
    # Test thresholds from min to max score
    min_score = np.min(scores)
    max_score = np.max(scores)
    thresholds = np.linspace(min_score, max_score, num_points)

    f1_scores = []
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        f1 = f1_score(ground_truth, predictions, zero_division=0)
        f1_scores.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return {
        "optimal_threshold": best_threshold,
        "optimal_f1": best_f1,
        "all_thresholds": thresholds,
        "all_f1_scores": f1_scores,
    }


def calculate_weighted_f1(ground_truth, predictions):
    """
    Calculate weighted F1-score taking into account class imbalance.

    Args:
        ground_truth: Array of ground truth labels (0/1)
        predictions: Array of binary predictions (0/1)

    Returns:
        float: Weighted F1-score
    """
    return f1_score(ground_truth, predictions, average="weighted", zero_division=0)


def run_advanced_evaluation(
    csv_path,
    annotations_dir,
    detection_column,
    output_dir,
    duration=10.0,
    optimize_threshold=True,
    excluded_classes=None,
):
    """
    Run complete advanced evaluation with threshold optimization and all plots.

    Args:
        csv_path: Path to CSV with predictions
        annotations_dir: Directory with annotation files
        detection_column: Column name for detection scores
        output_dir: Output directory for results
        duration: Segment duration in seconds
        optimize_threshold: Whether to optimize threshold by F1-score
        excluded_classes: Dict with classes to exclude from error analysis

    Returns:
        dict: Complete evaluation results
    """
    print("\nÉVALUATION AVANCÉE")
    print("=" * 60)

    # Use existing function to get segments and initial metrics
    # This will handle all the complex logic of matching files and creating segments
    segments_df, initial_metrics = analyze_detection_performance(
        csv_path, annotations_dir, detection_column, 0.001, None, duration
    )

    print(f"Segments created: {len(segments_df)}")

    # Extract arrays for advanced analysis
    ground_truth = segments_df["ground_truth"].values
    scores = segments_df["score"].values  # Raw scores for threshold optimization

    # Initial evaluation with low threshold (from existing analysis)
    initial_threshold = 0.001
    initial_predictions = segments_df["prediction"].values

    print(f"\nANALYSE INITIALE (seuil = {initial_threshold})")
    print("-" * 50)
    print(f"Positive segments (ground truth): {ground_truth.sum()}")
    print(f"Detected segments: {initial_predictions.sum()}")

    # Calculate advanced metrics for initial predictions
    advanced_initial_metrics = calculate_advanced_metrics(
        ground_truth, scores, initial_predictions
    )

    for key, value in advanced_initial_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key.replace('_', ' ').title()}: {value:.3f}")

    # Optimize threshold if requested
    if (
        optimize_threshold and len(np.unique(scores)) > 1
    ):  # Only if we have varied scores
        print("\nOPTIMISATION DU SEUIL")
        print("-" * 50)

        optimization_results = optimize_threshold_by_f1(ground_truth, scores)
        optimal_threshold = optimization_results["optimal_threshold"]
        optimal_f1 = optimization_results["optimal_f1"]

        print(f"Seuil optimal: {optimal_threshold:.6f}")
        print(f"F1-Score optimal: {optimal_f1:.3f}")

        # Evaluate with optimal threshold
        optimal_predictions = (scores >= optimal_threshold).astype(int)
        optimal_metrics = calculate_advanced_metrics(
            ground_truth, scores, optimal_predictions
        )

        print(f"\nMÉTRIQUES AVEC SEUIL OPTIMAL")
        print("-" * 50)
        for key, value in optimal_metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key.replace('_', ' ').title()}: {value:.3f}")

        # Use optimal threshold for final evaluation
        final_predictions = optimal_predictions
        final_metrics = optimal_metrics
        final_threshold = optimal_threshold
    else:
        print(
            "\nOptimisation du seuil ignorée (scores identiques ou option désactivée)"
        )
        final_predictions = initial_predictions
        final_metrics = advanced_initial_metrics
        final_threshold = initial_threshold
        optimization_results = None

    # Create all advanced visualizations
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Advanced plots
        plot_results = create_advanced_plots(
            ground_truth, final_predictions, scores, output_dir, detection_column
        )

        # Error analysis by class
        create_error_analysis_by_class(
            csv_path,
            segments_df,
            ground_truth,
            final_predictions,
            output_dir,
            detection_column,
            excluded_classes,
        )

        # Save detailed results
        segments_df["final_prediction"] = final_predictions
        segments_df["used_threshold"] = final_threshold

        results_path = os.path.join(
            output_dir, f"advanced_results_{detection_column}.csv"
        )
        segments_df.to_csv(results_path, index=False)
        print(f"\nRésultats détaillés: {results_path}")

        # Save metrics summary
        metrics_path = os.path.join(
            output_dir, f"advanced_metrics_{detection_column}.json"
        )

        # Convert numpy types to Python types for JSON serialization
        json_metrics = {}
        for key, value in final_metrics.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                json_metrics[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                json_metrics[key] = float(value)
            elif isinstance(value, np.ndarray):
                # Skip arrays as they can't be serialized to JSON
                continue
            elif value is None:
                json_metrics[key] = None
            elif isinstance(value, (int, float, str, bool)):
                json_metrics[key] = value
            else:
                # Skip other complex types
                continue

        json_metrics.update(
            {
                "used_threshold": float(final_threshold),
                "column_evaluated": detection_column,
                "total_segments": int(len(segments_df)),
                "positive_segments": int(ground_truth.sum()),
                "detected_segments": int(final_predictions.sum()),
            }
        )

        if optimization_results:
            json_metrics.update(
                {
                    "optimal_threshold": float(
                        plot_results.get("optimal_threshold", 0)
                    ),
                    "optimal_f1": float(plot_results.get("optimal_f1", 0)),
                    "roc_auc_plot": float(plot_results.get("roc_auc", 0)),
                    "pr_auc_plot": float(plot_results.get("pr_auc", 0)),
                }
            )

        with open(metrics_path, "w") as f:
            json.dump(json_metrics, f, indent=2)
        print(f"Métriques sauvegardées: {metrics_path}")

    return {
        "segments_df": segments_df,
        "metrics": final_metrics,
        "threshold_used": final_threshold,
        "optimization_results": optimization_results,
    }


if __name__ == "__main__":
    # Configuration using relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))

    # Use config from config_minimal.py if available
    try:
        import sys

        sys.path.append(os.path.join(parent_dir, "scripts", "config"))
        from config_minimal import load_config

        config = load_config()
        CSV_PATH = config["csv_file"]
        ANNOTATIONS_DIR = config["annotations_dir"]
        DETECTION_COLUMN = config["columns"][
            0
        ]  # Use first column (should be "Group_buzz")
        THRESHOLD = config["threshold"]
        OUTPUT_DIR = config["output_dir"]

        print(f"📁 Configuration loaded from config_minimal.py")
        print(f"   CSV: {CSV_PATH}")
        print(f"   Annotations: {ANNOTATIONS_DIR}")
        print(f"   Column: {DETECTION_COLUMN}")
        print(f"   Threshold: {THRESHOLD}")

    except Exception as e:
        print(f"Could not load config: {e}")
        print("Using default configuration...")

        # Fallback configuration
        CSV_PATH = os.path.join(parent_dir, "output_batch", "merged_results.csv")
        ANNOTATIONS_DIR = os.path.join(
            parent_dir, "data", "20240408_session_01_Tent", "SM05_T_annotées"
        )
        DETECTION_COLUMN = "Group_buzz"  # Column for buzz detection (as per config)
        THRESHOLD = 0.001  # Detection threshold
        OUTPUT_DIR = os.path.join(parent_dir, "output_batch", "evaluation_advanced")

    # Launch advanced analysis
    print("\n🚀 LANCEMENT DE L'ÉVALUATION AVANCÉE")
    print("=" * 60)

    results = run_advanced_evaluation(
        CSV_PATH,
        ANNOTATIONS_DIR,
        DETECTION_COLUMN,
        OUTPUT_DIR,
        duration=10.0,
        optimize_threshold=True,
    )

    print("\n✅ ÉVALUATION AVANCÉE TERMINÉE!")
    print("=" * 60)
    print(f"📊 Segments analysés: {len(results['segments_df'])}")
    print(f"🎯 Seuil utilisé: {results['threshold_used']:.6f}")

    # Print final metrics summary
    metrics = results["metrics"]
    print(f"\n📈 RÉSUMÉ DES MÉTRIQUES FINALES:")
    print("-" * 40)
    for key in ["precision", "recall", "f1_score", "weighted_f1", "accuracy"]:
        if key in metrics:
            print(f"{key.replace('_', ' ').title()}: {metrics[key]:.3f}")

    if results["optimization_results"]:
        opt = results["optimization_results"]
        print(f"\n🎯 Optimisation du seuil:")
        print(f"   F1 optimal: {opt['optimal_f1']:.3f}")
        print(f"   Seuil optimal: {opt['optimal_threshold']:.6f}")

    print(f"\n💾 Résultats sauvegardés dans: {OUTPUT_DIR}")
