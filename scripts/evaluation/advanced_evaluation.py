#!/usr/bin/env python3
"""
Advanced evaluation script for comparing multiple detection columns.
This script provides comprehensive analysis including false positive/negative breakdown.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_curve,
    auc,
    confusion_matrix,
)
import os
import argparse
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)


def load_ground_truth_data(ground_truth_dir):
    """
    Load ground truth data from annotation files.

    Args:
        ground_truth_dir: Directory containing ground truth annotation files

    Returns:
        dict: Dictionary mapping filenames to ground truth annotations
    """
    ground_truth = {}

    if not os.path.exists(ground_truth_dir):
        print(f"Warning: Ground truth directory not found: {ground_truth_dir}")
        return ground_truth

    for txt_file in os.listdir(ground_truth_dir):
        if txt_file.endswith(".txt"):
            file_path = os.path.join(ground_truth_dir, txt_file)
            annotations = []

            try:
                with open(file_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split("\t")
                            if len(parts) >= 2:
                                start_time = float(parts[0])
                                end_time = float(parts[1])
                                annotations.append((start_time, end_time))

                ground_truth[txt_file.replace(".txt", ".wav")] = annotations
            except Exception as e:
                print(f"Error reading {txt_file}: {e}")

    return ground_truth


def create_segment_labels(df, ground_truth_data, segment_duration=5.0):
    """
    Create binary labels for segments based on ground truth.

    Args:
        df: DataFrame with prediction data
        ground_truth_data: Dictionary with ground truth annotations
        segment_duration: Duration of each segment in seconds

    Returns:
        list: Binary labels for each segment
    """
    labels = []

    for _, row in df.iterrows():
        filename = row["name"]
        start_time = row["start"]
        end_time = start_time + segment_duration

        # Check if this segment overlaps with any ground truth annotation
        has_event = False
        if filename in ground_truth_data:
            for gt_start, gt_end in ground_truth_data[filename]:
                # Check for overlap
                if not (end_time <= gt_start or start_time >= gt_end):
                    has_event = True
                    break

        labels.append(1 if has_event else 0)

    return labels


def analyze_false_positives_negatives(
    df, ground_truth_labels, predictions, column_name
):
    """
    Analyze false positives and false negatives in detail.

    Args:
        df: DataFrame with prediction data
        ground_truth_labels: True labels
        predictions: Predicted labels
        column_name: Name of the detection column

    Returns:
        dict: Detailed analysis of errors
    """
    # Create confusion matrix
    cm = confusion_matrix(ground_truth_labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    # Find false positives and false negatives
    false_positives = []
    false_negatives = []

    for i, (gt, pred) in enumerate(zip(ground_truth_labels, predictions)):
        if gt == 0 and pred == 1:  # False positive
            false_positives.append(
                {
                    "index": i,
                    "filename": df.iloc[i]["name"],
                    "start_time": df.iloc[i]["start"],
                    "score": df.iloc[i][column_name],
                    "dominant_class": get_dominant_class(df.iloc[i]),
                }
            )
        elif gt == 1 and pred == 0:  # False negative
            false_negatives.append(
                {
                    "index": i,
                    "filename": df.iloc[i]["name"],
                    "start_time": df.iloc[i]["start"],
                    "score": df.iloc[i][column_name],
                    "dominant_class": get_dominant_class(df.iloc[i]),
                }
            )

    return {
        "confusion_matrix": cm,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "false_positive_details": false_positives,
        "false_negative_details": false_negatives,
    }


def get_dominant_class(row):
    """
    Get the dominant class for a given row.

    Args:
        row: DataFrame row

    Returns:
        str: Name of the dominant class
    """
    class_columns = [col for col in row.index if col.startswith("tag_")]
    if not class_columns:
        return "Unknown"

    class_scores = {col: row[col] for col in class_columns}
    dominant_class = max(class_scores, key=class_scores.get)

    return dominant_class.replace("tag_", "")


def create_error_analysis_table(error_analysis, output_dir, column_name):
    """
    Create detailed tables for false positives and false negatives.

    Args:
        error_analysis: Dictionary with error analysis
        output_dir: Output directory
        column_name: Name of the detection column
    """
    # False Positives Table
    if error_analysis["false_positive_details"]:
        fp_df = pd.DataFrame(error_analysis["false_positive_details"])
        fp_df = fp_df.sort_values("score", ascending=False)
        fp_df["error_type"] = "False Positive"

        # Group by dominant class
        fp_by_class = (
            fp_df.groupby("dominant_class")
            .agg({"score": ["count", "mean", "std"], "filename": "count"})
            .round(3)
        )

        fp_by_class.columns = ["Count", "Mean_Score", "Std_Score", "File_Count"]
        fp_by_class = fp_by_class.sort_values("Count", ascending=False)

        # Save detailed FP table
        fp_df.to_csv(
            os.path.join(output_dir, f"false_positives_{column_name}.csv"), index=False
        )
        fp_by_class.to_csv(
            os.path.join(output_dir, f"false_positives_by_class_{column_name}.csv")
        )

        print(f"False Positives Analysis for {column_name}:")
        print(f"Total False Positives: {len(fp_df)}")
        print("By Dominant Class:")
        print(fp_by_class)
        print()

    # False Negatives Table
    if error_analysis["false_negative_details"]:
        fn_df = pd.DataFrame(error_analysis["false_negative_details"])
        fn_df = fn_df.sort_values("score", ascending=True)
        fn_df["error_type"] = "False Negative"

        # Group by dominant class
        fn_by_class = (
            fn_df.groupby("dominant_class")
            .agg({"score": ["count", "mean", "std"], "filename": "count"})
            .round(3)
        )

        fn_by_class.columns = ["Count", "Mean_Score", "Std_Score", "File_Count"]
        fn_by_class = fn_by_class.sort_values("Count", ascending=False)

        # Save detailed FN table
        fn_df.to_csv(
            os.path.join(output_dir, f"false_negatives_{column_name}.csv"), index=False
        )
        fn_by_class.to_csv(
            os.path.join(output_dir, f"false_negatives_by_class_{column_name}.csv")
        )

        print(f"False Negatives Analysis for {column_name}:")
        print(f"Total False Negatives: {len(fn_df)}")
        print("By Dominant Class:")
        print(fn_by_class)
        print()

    # Combined summary table
    summary_data = []

    # Add false positives by class
    if error_analysis["false_positive_details"]:
        fp_df = pd.DataFrame(error_analysis["false_positive_details"])
        fp_summary = (
            fp_df.groupby("dominant_class").agg({"score": ["count", "mean"]}).round(3)
        )
        fp_summary.columns = ["FP_Count", "FP_Mean_Score"]

        for class_name, row in fp_summary.iterrows():
            summary_data.append(
                {
                    "Class": class_name,
                    "False_Positives": row["FP_Count"],
                    "FP_Mean_Score": row["FP_Mean_Score"],
                    "False_Negatives": 0,
                    "FN_Mean_Score": 0,
                }
            )

    # Add false negatives by class
    if error_analysis["false_negative_details"]:
        fn_df = pd.DataFrame(error_analysis["false_negative_details"])
        fn_summary = (
            fn_df.groupby("dominant_class").agg({"score": ["count", "mean"]}).round(3)
        )
        fn_summary.columns = ["FN_Count", "FN_Mean_Score"]

        for class_name, row in fn_summary.iterrows():
            # Find existing entry or create new one
            existing = next(
                (item for item in summary_data if item["Class"] == class_name), None
            )
            if existing:
                existing["False_Negatives"] = row["FN_Count"]
                existing["FN_Mean_Score"] = row["FN_Mean_Score"]
            else:
                summary_data.append(
                    {
                        "Class": class_name,
                        "False_Positives": 0,
                        "FP_Mean_Score": 0,
                        "False_Negatives": row["FN_Count"],
                        "FN_Mean_Score": row["FN_Mean_Score"],
                    }
                )

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df["Total_Errors"] = (
            summary_df["False_Positives"] + summary_df["False_Negatives"]
        )
        summary_df = summary_df.sort_values("Total_Errors", ascending=False)

        summary_df.to_csv(
            os.path.join(output_dir, f"error_summary_{column_name}.csv"), index=False
        )

        print(f"Error Summary Table for {column_name}:")
        print(summary_df.to_string(index=False))
        print()


def advanced_evaluation(csv_file, ground_truth_dir, output_dir, columns=None):
    """
    Advanced evaluation for multiple detection columns.

    Args:
        csv_file: CSV file with predictions
        ground_truth_dir: Directory with ground truth annotations
        output_dir: Output directory
        columns: List of columns to evaluate
    """
    print("Advanced Multi-Column Evaluation")
    print("=" * 50)

    if columns is None:
        columns = ["tag_Buzz", "tag_Insect", "buzz"]

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_file)
    ground_truth_data = load_ground_truth_data(ground_truth_dir)

    # Create ground truth labels for segments
    ground_truth_labels = create_segment_labels(df, ground_truth_data)

    results = {}

    for col in columns:
        if col in df.columns:
            print(f"Evaluating column: {col}")

            # Use threshold of 0.5 for binary classification
            threshold = 0.5
            predictions = (df[col] > threshold).astype(int)

            # Calculate basic metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                ground_truth_labels, predictions, average="binary", zero_division=0
            )

            # Detailed error analysis
            error_analysis = analyze_false_positives_negatives(
                df, ground_truth_labels, predictions, col
            )

            # Create error analysis tables
            create_error_analysis_table(error_analysis, output_dir, col)

            results[col] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "predictions": predictions,
                "ground_truth": ground_truth_labels,
                "scores": df[col].values,
                "error_analysis": error_analysis,
            }
        else:
            print(f"Warning: Column {col} not found in data")

    # Create comparative visualization
    if results:
        create_comparative_plots(results, output_dir)

    # Save summary metrics
    summary = pd.DataFrame(
        [
            {
                "column": col,
                "precision": data["precision"],
                "recall": data["recall"],
                "f1_score": data["f1_score"],
                "true_positives": data["error_analysis"]["true_positives"],
                "false_positives": data["error_analysis"]["false_positives"],
                "true_negatives": data["error_analysis"]["true_negatives"],
                "false_negatives": data["error_analysis"]["false_negatives"],
            }
            for col, data in results.items()
        ]
    )

    summary.to_csv(os.path.join(output_dir, "advanced_metrics.csv"), index=False)

    print("Advanced evaluation completed")
    print(f"Results saved to: {output_dir}")

    return results


def create_comparative_plots(results, output_dir):
    """
    Create comparative plots for all evaluated columns.

    Args:
        results: Dictionary with evaluation results
        output_dir: Output directory
    """
    if not results:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Metrics comparison
    metrics = ["precision", "recall", "f1_score"]
    colors = ["skyblue", "lightgreen", "lightcoral"]

    for i, metric in enumerate(metrics):
        ax = axes[0, i] if i < 2 else axes[1, 0]
        values = [results[col][metric] for col in results.keys()]
        bars = ax.bar(results.keys(), values, color=colors[i], alpha=0.7)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # ROC curves
    ax = axes[1, 1]
    for col in results.keys():
        fpr, tpr, _ = roc_curve(results[col]["ground_truth"], results[col]["scores"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{col} (AUC = {roc_auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "advanced_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Error analysis plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    error_data = []
    for col in results.keys():
        error_analysis = results[col]["error_analysis"]
        error_data.append(
            {
                "Column": col,
                "False Positives": error_analysis["false_positives"],
                "False Negatives": error_analysis["false_negatives"],
            }
        )

    error_df = pd.DataFrame(error_data)
    x = np.arange(len(error_df))
    width = 0.35

    ax.bar(
        x - width / 2,
        error_df["False Positives"],
        width,
        label="False Positives",
        color="red",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        error_df["False Negatives"],
        width,
        label="False Negatives",
        color="orange",
        alpha=0.7,
    )

    ax.set_xlabel("Detection Columns")
    ax.set_ylabel("Number of Errors")
    ax.set_title("Error Analysis by Detection Column")
    ax.set_xticks(x)
    ax.set_xticklabels(error_df["Column"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "error_analysis.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Advanced multi-column evaluation")
    parser.add_argument("--csv", required=True, help="CSV file with predictions")
    parser.add_argument("--ground_truth", required=True, help="Ground truth directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["tag_Buzz", "tag_Insect", "buzz"],
        help="Columns to evaluate",
    )

    args = parser.parse_args()

    advanced_evaluation(args.csv, args.ground_truth, args.output, args.columns)


if __name__ == "__main__":
    main()
