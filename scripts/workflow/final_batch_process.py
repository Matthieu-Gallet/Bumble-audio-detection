#!/usr/bin/env python3
"""
Final script to automate complete processing of all directories with advanced evaluation.

This script combines batch processing capabilities with comprehensive evaluation features:

PROCESSING FEATURES:
- Automatic discovery of data and annotation directories
- Smart processing with existing file detection (avoids reprocessing)
- Parallel processing of multiple data sessions
- Intelligent CSV merging with row count validation

EVALUATION FEATURES:
- Standard evaluation with fixed threshold (0.5)
- Optimal threshold detection that maximizes F1-score
- Detailed false positive/negative analysis by class
- Class-based error breakdown and visualization
- Comprehensive error tables and summaries
- Performance comparison plots (default vs optimal)

ANALYSIS MODES:
- Individual: Analyze each session separately
- Combined: Analyze all sessions together (traditional approach)
- Both: Run both individual and combined analysis

OUTPUT FILES:
- Individual session CSVs: output_batch/[session]/indices_[session].csv
- Merged results: output_batch/merged_results.csv
- Individual session analysis: output_batch/eval_individual_[session]/
- Combined analysis: output_batch/eval_[column]/
- Cross-session comparison: output_batch/cross_session_analysis/
- Summary reports: evaluation_summary.csv, optimal_evaluation_summary.csv
- Error analysis: false_positives_detailed_[column].csv, false_negatives_detailed_[column].csv
- Visualizations: threshold_comparison.png, error_analysis_detailed_[column].png

USAGE:
    # Run both individual and combined analysis (default)
    python final_batch_process.py

    # Run only individual session analysis
    python final_batch_process.py --analysis-mode individual

    # Run only combined analysis
    python final_batch_process.py --analysis-mode combined

    # Analyze specific sessions only
    python final_batch_process.py --sessions 20240408_session_01_Tent_SM05_T 20240612_session_02_Tent_SM06_T

    # Skip data processing and go directly to evaluation
    python final_batch_process.py --skip-processing

    # Evaluate specific columns only
    python final_batch_process.py --columns tag_Buzz buzz

The script will automatically:
1. Find all data directories in ../data/
2. Process each directory (or skip if already done)
3. Merge all results
4. Run comprehensive evaluation based on selected mode:
   - Individual: Per-session analysis with cross-session comparison
   - Combined: Traditional all-data-together analysis
   - Both: Complete analysis with maximum insights
5. Generate detailed error reports and visualizations
"""

import os
import subprocess
import pandas as pd
import glob
from pathlib import Path
import time
import shutil
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import argparse
import yaml

# Global configuration - use dynamic paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# Add config directory to path
config_dir = os.path.join(current_dir, "..", "config")
sys.path.insert(0, config_dir)

try:
    from config_loader import WorkflowConfig

    config = WorkflowConfig()

    # Use configuration values
    BASE_DATA_PATH = config.data_path
    PYTHON_PATH = config.python_path
    PROCESS_SCRIPT = config.process_script
    OUTPUT_BASE = config.output_base
    SEGMENT_LENGTH = config.segment_length

except ImportError:
    print("⚠️  Configuration system not available, using default values")
    # Fallback to original configuration
    BASE_DATA_PATH = os.path.join(project_root, "data")
    PYTHON_PATH = os.path.join(project_root, ".venv", "bin", "python")
    PROCESS_SCRIPT = os.path.join(project_root, "process.py")
    OUTPUT_BASE = os.path.join(project_root, "output_batch")
    SEGMENT_LENGTH = 5
    config = None


def find_data_directories(base_data_path: str) -> list:
    """Find all data directories."""
    data_dirs = []

    for item in os.listdir(base_data_path):
        item_path = os.path.join(base_data_path, item)

        if os.path.isdir(item_path) and not item.endswith("_annotées"):
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)

                if os.path.isdir(subitem_path) and not subitem.endswith("_annotées"):
                    audio_files = glob.glob(os.path.join(subitem_path, "*.wav"))
                    if audio_files:
                        data_dirs.append(
                            {
                                "session": item,
                                "subdirectory": subitem,
                                "full_path": subitem_path,
                                "audio_count": len(audio_files),
                            }
                        )

    return data_dirs


def find_annotation_directories(base_data_path: str) -> list:
    """Find all annotation directories."""
    annotation_dirs = []

    for item in os.listdir(base_data_path):
        item_path = os.path.join(base_data_path, item)

        if os.path.isdir(item_path):
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)

                if os.path.isdir(subitem_path) and subitem.endswith("_annotées"):
                    annotation_files = glob.glob(os.path.join(subitem_path, "*.txt"))
                    if annotation_files:
                        annotation_dirs.append(
                            {
                                "session": item,
                                "subdirectory": subitem,
                                "full_path": subitem_path,
                                "annotation_count": len(annotation_files),
                            }
                        )

    return annotation_dirs


def run_process_for_directory(data_dir: dict) -> str:
    """Process a data directory."""
    session_name = f"{data_dir['session']}_{data_dir['subdirectory']}"
    output_dir = os.path.join(OUTPUT_BASE, session_name)
    csv_file = os.path.join(output_dir, f"indices_{session_name}.csv")

    # Check if processing is already done
    if os.path.exists(csv_file):
        print(f"⏭️  Already processed: {session_name}")
        return csv_file

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        PYTHON_PATH,
        PROCESS_SCRIPT,
        "--data_path",
        data_dir["full_path"],
        "--save_path",
        output_dir,
        "--name",
        session_name,
        "--audio_format",
        "wav",
        "--l",
        str(SEGMENT_LENGTH),
    ]

    print(f"Processing: {session_name}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        if result.returncode == 0:
            if os.path.exists(csv_file):
                print(f"✅ Success: {csv_file}")
                return csv_file
            else:
                print(f"❌ CSV file not found")
                return None
        else:
            print(f"❌ Error: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def merge_csv_files(csv_files: list) -> str:
    """Merge CSV files."""
    merged_path = os.path.join(OUTPUT_BASE, "merged_results.csv")

    # Check if merged file already exists and is up to date
    if os.path.exists(merged_path):
        try:
            existing_df = pd.read_csv(merged_path)

            # Check if all individual CSV files exist and have content
            total_expected_rows = 0
            for csv_file in csv_files:
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    total_expected_rows += len(df)

            # If the merged file has the expected number of rows (or close), skip merging
            if len(existing_df) >= total_expected_rows * 0.95:  # Allow 5% tolerance
                print(
                    f"⏭️  Merged file already exists: {merged_path} ({len(existing_df)} rows)"
                )
                return merged_path
        except Exception as e:
            print(f"⚠️  Error checking existing merged file: {e}")

    print(f"🔄 Merging {len(csv_files)} files...")

    all_dfs = []
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            all_dfs.append(df)

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.to_csv(merged_path, index=False)
        print(f"✅ Merged file: {merged_path} ({len(merged_df)} rows)")
        return merged_path

    return None


def create_combined_annotations(annotation_dirs: list) -> str:
    """Create a temporary directory with all annotations."""
    temp_dir = os.path.join(OUTPUT_BASE, "temp_annotations")
    os.makedirs(temp_dir, exist_ok=True)

    print("🎯 Combining annotations...")

    for ann_dir in annotation_dirs:
        for txt_file in Path(ann_dir["full_path"]).glob("*.txt"):
            # Copy file to temporary directory
            dest_path = os.path.join(temp_dir, txt_file.name)
            shutil.copy2(txt_file, dest_path)

    annotation_files = glob.glob(os.path.join(temp_dir, "*.txt"))
    print(f"✅ {len(annotation_files)} annotation files combined")

    return temp_dir


def run_evaluation(merged_csv: str, annotations_dir: str):
    """Launch comprehensive evaluation with advanced analysis."""
    print("📊 Evaluation in progress...")

    # Add project root to Python path
    sys.path.insert(0, project_root)

    try:
        # Add the scripts directory to path and import
        scripts_dir = os.path.dirname(current_dir)
        sys.path.insert(0, scripts_dir)
        from evaluation.evaluate_detection import analyze_detection_performance

        # Columns to evaluate
        columns = ["tag_Buzz", "tag_Insect", "tag_Bird", "buzz", "biophony"]

        # Standard evaluation results
        results = []

        # Advanced evaluation results with optimal thresholds
        optimal_results = []

        for col in columns:
            print(f"  🔍 Standard evaluation: {col}")

            eval_dir = os.path.join(OUTPUT_BASE, f"eval_{col}")

            try:
                # Standard evaluation with threshold 0.5
                segments_df, metrics = analyze_detection_performance(
                    merged_csv, annotations_dir, col, 0.5, eval_dir, duration=5.0
                )

                results.append(
                    {
                        "column": col,
                        "f1_score": metrics["f1_score"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "accuracy": metrics["accuracy"],
                    }
                )

                # Advanced evaluation with optimal threshold
                print(f"  🎯 Advanced evaluation: {col}")
                optimal_result = run_analysis_with_optimal_threshold(
                    merged_csv, annotations_dir, col, eval_dir
                )

                if optimal_result:
                    optimal_results.append(
                        {
                            "column": col,
                            "default_f1": metrics["f1_score"],
                            "default_precision": metrics["precision"],
                            "default_recall": metrics["recall"],
                            "optimal_threshold": optimal_result["optimal_threshold"],
                            "optimal_f1": optimal_result["optimal_metrics"]["f1_score"],
                            "optimal_precision": optimal_result["optimal_metrics"][
                                "precision"
                            ],
                            "optimal_recall": optimal_result["optimal_metrics"][
                                "recall"
                            ],
                            "improvement": optimal_result["optimal_metrics"]["f1_score"]
                            - metrics["f1_score"],
                            "false_positives": optimal_result["error_analysis"][
                                "false_positives_count"
                            ],
                            "false_negatives": optimal_result["error_analysis"][
                                "false_negatives_count"
                            ],
                        }
                    )

            except Exception as e:
                print(f"    ❌ Error: {e}")

        # Create standard summary
        if results:
            summary_df = pd.DataFrame(results)
            summary_path = os.path.join(OUTPUT_BASE, "evaluation_summary.csv")
            summary_df.to_csv(summary_path, index=False)

            print(f"\n📊 STANDARD EVALUATION SUMMARY:")
            print(summary_df.to_string(index=False, float_format="%.3f"))

            best_col = summary_df.loc[summary_df["f1_score"].idxmax()]
            print(
                f"\n🏆 Best column (standard): {best_col['column']} (F1: {best_col['f1_score']:.3f})"
            )

        # Create optimal threshold summary
        if optimal_results:
            optimal_df = pd.DataFrame(optimal_results)
            optimal_path = os.path.join(OUTPUT_BASE, "optimal_evaluation_summary.csv")
            optimal_df.to_csv(optimal_path, index=False)

            print(f"\n🎯 OPTIMAL THRESHOLD EVALUATION SUMMARY:")
            print(
                optimal_df[
                    [
                        "column",
                        "optimal_threshold",
                        "default_f1",
                        "optimal_f1",
                        "improvement",
                    ]
                ].to_string(index=False, float_format="%.3f")
            )

            best_optimal = optimal_df.loc[optimal_df["optimal_f1"].idxmax()]
            print(
                f"\n🏆 Best column (optimal): {best_optimal['column']} (F1: {best_optimal['optimal_f1']:.3f}, Threshold: {best_optimal['optimal_threshold']:.3f})"
            )

            # Create comparison visualization
            create_threshold_comparison_plot(optimal_df, OUTPUT_BASE)

    except ImportError as e:
        print(f"❌ Import error: {e}")


def create_threshold_comparison_plot(optimal_df: pd.DataFrame, output_dir: str):
    """Create visualization comparing default vs optimal thresholds."""
    plt.figure(figsize=(15, 10))

    x = np.arange(len(optimal_df))
    width = 0.35

    # F1-Score comparison
    plt.subplot(2, 2, 1)
    plt.bar(
        [i - width / 2 for i in x],
        optimal_df["default_f1"],
        width,
        label="Threshold 0.5",
        alpha=0.7,
        color="lightblue",
    )
    plt.bar(
        [i + width / 2 for i in x],
        optimal_df["optimal_f1"],
        width,
        label="Optimal threshold",
        alpha=0.7,
        color="darkblue",
    )
    plt.xlabel("Columns")
    plt.ylabel("F1-Score")
    plt.title("F1-Score Comparison")
    plt.xticks(x, optimal_df["column"], rotation=45)
    plt.legend()

    # Precision comparison
    plt.subplot(2, 2, 2)
    plt.bar(
        [i - width / 2 for i in x],
        optimal_df["default_precision"],
        width,
        label="Threshold 0.5",
        alpha=0.7,
        color="lightgreen",
    )
    plt.bar(
        [i + width / 2 for i in x],
        optimal_df["optimal_precision"],
        width,
        label="Optimal threshold",
        alpha=0.7,
        color="darkgreen",
    )
    plt.xlabel("Columns")
    plt.ylabel("Precision")
    plt.title("Precision Comparison")
    plt.xticks(x, optimal_df["column"], rotation=45)
    plt.legend()

    # Recall comparison
    plt.subplot(2, 2, 3)
    plt.bar(
        [i - width / 2 for i in x],
        optimal_df["default_recall"],
        width,
        label="Threshold 0.5",
        alpha=0.7,
        color="lightcoral",
    )
    plt.bar(
        [i + width / 2 for i in x],
        optimal_df["optimal_recall"],
        width,
        label="Optimal threshold",
        alpha=0.7,
        color="darkred",
    )
    plt.xlabel("Columns")
    plt.ylabel("Recall")
    plt.title("Recall Comparison")
    plt.xticks(x, optimal_df["column"], rotation=45)
    plt.legend()

    # Improvement
    plt.subplot(2, 2, 4)
    colors = ["green" if imp > 0 else "red" for imp in optimal_df["improvement"]]
    plt.bar(x, optimal_df["improvement"], color=colors, alpha=0.7)
    plt.xlabel("Columns")
    plt.ylabel("F1-Score Improvement")
    plt.title("Improvement with Optimal Threshold")
    plt.xticks(x, optimal_df["column"], rotation=45)
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "threshold_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"✅ Threshold comparison plot saved")


def find_optimal_threshold(
    segments_df: pd.DataFrame, score_column: str = "score"
) -> dict:
    """
    Find optimal threshold that maximizes F1-score.

    Args:
        segments_df: DataFrame with ground_truth and score
        score_column: Name of column containing scores

    Returns:
        Dictionary with optimal threshold and metrics
    """
    thresholds = np.linspace(0, 1, 101)
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = None

    for threshold in thresholds:
        y_pred = (segments_df[score_column] > threshold).astype(int)
        y_true = segments_df["ground_truth"]

        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

                # Calculate complete metrics
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                best_metrics = {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "accuracy": accuracy,
                    "specificity": specificity,
                    "true_positives": tp,
                    "false_positives": fp,
                    "true_negatives": tn,
                    "false_negatives": fn,
                }

    return {"optimal_threshold": best_threshold, "optimal_metrics": best_metrics}


def analyze_false_positives_classes(merged_csv: str, segments_df: pd.DataFrame) -> dict:
    """
    Analyze classes that contribute most to false positives.

    Args:
        merged_csv: Path to the merged CSV file
        segments_df: DataFrame with evaluation results

    Returns:
        Dictionary with false positive analysis
    """
    print("Analyzing classes causing false positives...")

    # Load full CSV data
    df_full = pd.read_csv(merged_csv)

    # Identify false positives
    false_positives = segments_df[
        (segments_df["ground_truth"] == 0) & (segments_df["prediction"] == 1)
    ]

    if len(false_positives) == 0:
        return {"message": "No false positives found"}

    # Columns of different classes to analyze
    class_columns = [col for col in df_full.columns if col.startswith("tag_")]

    # Analyze average scores for false positives
    fp_analysis = {}

    for fp_idx, fp_row in false_positives.iterrows():
        # Find corresponding row in full CSV
        matching_rows = df_full[
            (df_full["name"] == fp_row["filename"])
            & (df_full["start"] == fp_row["start_time"])
        ]

        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]

            # Identify top 3 classes with highest scores
            class_scores = {}
            for col in class_columns:
                if col in row:
                    class_scores[col] = row[col]

            # Sort by descending score
            sorted_classes = sorted(
                class_scores.items(), key=lambda x: x[1], reverse=True
            )
            top_3_classes = sorted_classes[:3]

            for class_name, score in top_3_classes:
                if class_name not in fp_analysis:
                    fp_analysis[class_name] = []
                fp_analysis[class_name].append(score)

    # Calculate statistics
    fp_stats = {}
    for class_name, scores in fp_analysis.items():
        fp_stats[class_name] = {
            "count": len(scores),
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "percentage": len(scores) / len(false_positives) * 100,
        }

    # Sort by frequency
    sorted_stats = sorted(fp_stats.items(), key=lambda x: x[1]["count"], reverse=True)

    print(f"Top 5 classes causing false positives:")
    for i, (class_name, stats) in enumerate(sorted_stats[:5], 1):
        print(
            f"  {i}. {class_name}: {stats['count']} occurrences ({stats['percentage']:.1f}%) - Mean score: {stats['mean_score']:.3f}"
        )

    return {
        "false_positives_count": len(false_positives),
        "class_analysis": fp_stats,
        "top_contributing_classes": sorted_stats[:5],
    }


def create_detailed_error_analysis_table(
    merged_csv: str, segments_df: pd.DataFrame, detection_column: str, output_dir: str
):
    """
    Create comprehensive error analysis tables for false positives and false negatives.

    Args:
        merged_csv: Path to merged CSV file
        segments_df: DataFrame with evaluation results
        detection_column: Detection column being analyzed
        output_dir: Output directory for tables
    """
    print(f"Creating detailed error analysis tables for {detection_column}...")

    # Load full CSV data
    df_full = pd.read_csv(merged_csv)

    # Get all tag columns for class analysis
    tag_columns = [col for col in df_full.columns if col.startswith("tag_")]

    # Initialize error data structures
    false_positives_data = []
    false_negatives_data = []

    # Process each segment
    for idx, segment in segments_df.iterrows():
        # Find corresponding row in full CSV
        matching_rows = df_full[
            (df_full["name"] == segment["filename"])
            & (df_full["start"] == segment["start_time"])
        ]

        if len(matching_rows) == 0:
            continue

        row = matching_rows.iloc[0]

        # Get class scores for this segment
        class_scores = {}
        for col in tag_columns:
            if col in row:
                class_scores[col] = row[col]

        # Find dominant class
        if class_scores:
            dominant_class = max(class_scores, key=class_scores.get)
            dominant_score = class_scores[dominant_class]

            # Get top 3 classes
            sorted_classes = sorted(
                class_scores.items(), key=lambda x: x[1], reverse=True
            )
            top_3_classes = sorted_classes[:3]
        else:
            dominant_class = "Unknown"
            dominant_score = 0
            top_3_classes = []

        # Check for false positives
        if segment["ground_truth"] == 0 and segment["prediction"] == 1:
            false_positives_data.append(
                {
                    "filename": segment["filename"],
                    "start_time": segment["start_time"],
                    "detection_score": segment["score"],
                    "dominant_class": dominant_class.replace("tag_", ""),
                    "dominant_score": dominant_score,
                    "class_1": (
                        top_3_classes[0][0].replace("tag_", "")
                        if len(top_3_classes) > 0
                        else "None"
                    ),
                    "score_1": top_3_classes[0][1] if len(top_3_classes) > 0 else 0,
                    "class_2": (
                        top_3_classes[1][0].replace("tag_", "")
                        if len(top_3_classes) > 1
                        else "None"
                    ),
                    "score_2": top_3_classes[1][1] if len(top_3_classes) > 1 else 0,
                    "class_3": (
                        top_3_classes[2][0].replace("tag_", "")
                        if len(top_3_classes) > 2
                        else "None"
                    ),
                    "score_3": top_3_classes[2][1] if len(top_3_classes) > 2 else 0,
                    "error_type": "False Positive",
                }
            )

        # Check for false negatives
        elif segment["ground_truth"] == 1 and segment["prediction"] == 0:
            false_negatives_data.append(
                {
                    "filename": segment["filename"],
                    "start_time": segment["start_time"],
                    "detection_score": segment["score"],
                    "dominant_class": dominant_class.replace("tag_", ""),
                    "dominant_score": dominant_score,
                    "class_1": (
                        top_3_classes[0][0].replace("tag_", "")
                        if len(top_3_classes) > 0
                        else "None"
                    ),
                    "score_1": top_3_classes[0][1] if len(top_3_classes) > 0 else 0,
                    "class_2": (
                        top_3_classes[1][0].replace("tag_", "")
                        if len(top_3_classes) > 1
                        else "None"
                    ),
                    "score_2": top_3_classes[1][1] if len(top_3_classes) > 1 else 0,
                    "class_3": (
                        top_3_classes[2][0].replace("tag_", "")
                        if len(top_3_classes) > 2
                        else "None"
                    ),
                    "score_3": top_3_classes[2][1] if len(top_3_classes) > 2 else 0,
                    "error_type": "False Negative",
                }
            )

    # Create False Positives DataFrame and analysis
    if false_positives_data:
        fp_df = pd.DataFrame(false_positives_data)
        fp_df = fp_df.sort_values("detection_score", ascending=False)

        # Save detailed FP table
        fp_df.to_csv(
            os.path.join(
                output_dir, f"false_positives_detailed_{detection_column}.csv"
            ),
            index=False,
        )

        # Create FP summary by dominant class
        fp_summary = (
            fp_df.groupby("dominant_class")
            .agg(
                {
                    "detection_score": ["count", "mean", "std", "min", "max"],
                    "dominant_score": ["mean", "std"],
                    "filename": "nunique",
                }
            )
            .round(4)
        )

        fp_summary.columns = [
            "FP_Count",
            "Det_Mean",
            "Det_Std",
            "Det_Min",
            "Det_Max",
            "Dom_Mean",
            "Dom_Std",
            "Unique_Files",
        ]
        fp_summary = fp_summary.sort_values("FP_Count", ascending=False)

        # Save FP summary
        fp_summary.to_csv(
            os.path.join(output_dir, f"false_positives_summary_{detection_column}.csv")
        )

        print(f"\nFALSE POSITIVES ANALYSIS - {detection_column}")
        print("=" * 60)
        print(f"Total False Positives: {len(fp_df)}")
        print(f"Affected Files: {fp_df['filename'].nunique()}")

    # Create False Negatives DataFrame and analysis
    if false_negatives_data:
        fn_df = pd.DataFrame(false_negatives_data)
        fn_df = fn_df.sort_values("detection_score", ascending=True)

        # Save detailed FN table
        fn_df.to_csv(
            os.path.join(
                output_dir, f"false_negatives_detailed_{detection_column}.csv"
            ),
            index=False,
        )

        # Create FN summary by dominant class
        fn_summary = (
            fn_df.groupby("dominant_class")
            .agg(
                {
                    "detection_score": ["count", "mean", "std", "min", "max"],
                    "dominant_score": ["mean", "std"],
                    "filename": "nunique",
                }
            )
            .round(4)
        )

        fn_summary.columns = [
            "FN_Count",
            "Det_Mean",
            "Det_Std",
            "Det_Min",
            "Det_Max",
            "Dom_Mean",
            "Dom_Std",
            "Unique_Files",
        ]
        fn_summary = fn_summary.sort_values("FN_Count", ascending=False)

        # Save FN summary
        fn_summary.to_csv(
            os.path.join(output_dir, f"false_negatives_summary_{detection_column}.csv")
        )

        print(f"\nFALSE NEGATIVES ANALYSIS - {detection_column}")
        print("=" * 60)
        print(f"Total False Negatives: {len(fn_df)}")
        print(f"Affected Files: {fn_df['filename'].nunique()}")

    return {
        "false_positives_count": len(false_positives_data),
        "false_negatives_count": len(false_negatives_data),
        "false_positives_data": false_positives_data,
        "false_negatives_data": false_negatives_data,
    }


def create_combined_error_summary(
    fp_data: list, fn_data: list, detection_column: str, output_dir: str
):
    """
    Create a combined summary of all errors by class.

    Args:
        fp_data: False positives data
        fn_data: False negatives data
        detection_column: Detection column name
        output_dir: Output directory
    """
    print(f"Creating combined error summary for {detection_column}...")

    # Combine all error data
    all_errors = []

    for fp in fp_data:
        all_errors.append(
            {
                "dominant_class": fp["dominant_class"],
                "error_type": "False Positive",
                "detection_score": fp["detection_score"],
                "dominant_score": fp["dominant_score"],
            }
        )

    for fn in fn_data:
        all_errors.append(
            {
                "dominant_class": fn["dominant_class"],
                "error_type": "False Negative",
                "detection_score": fn["detection_score"],
                "dominant_score": fn["dominant_score"],
            }
        )

    if not all_errors:
        return

    error_df = pd.DataFrame(all_errors)

    # Create summary by class and error type
    summary = (
        error_df.groupby(["dominant_class", "error_type"])
        .agg(
            {
                "detection_score": ["count", "mean", "std"],
                "dominant_score": ["mean", "std"],
            }
        )
        .round(4)
    )

    summary.columns = ["Count", "Det_Mean", "Det_Std", "Dom_Mean", "Dom_Std"]
    summary = summary.reset_index()

    # Save combined summary
    summary.to_csv(
        os.path.join(output_dir, f"combined_error_summary_{detection_column}.csv"),
        index=False,
    )

    print(f"✅ Combined error summary saved for {detection_column}")


def create_error_visualization(
    fp_data: list, fn_data: list, detection_column: str, output_dir: str
):
    """
    Create visualizations for error analysis.

    Args:
        fp_data: False positives data
        fn_data: False negatives data
        detection_column: Detection column name
        output_dir: Output directory
    """
    if not fp_data and not fn_data:
        return

    print(f"Creating error visualizations for {detection_column}...")

    plt.figure(figsize=(15, 10))

    # Plot 1: False Positives by class
    if fp_data:
        fp_df = pd.DataFrame(fp_data)
        fp_class_counts = fp_df["dominant_class"].value_counts().head(10)

        plt.subplot(2, 2, 1)
        fp_class_counts.plot(kind="bar", color="red", alpha=0.7)
        plt.title(f"Top 10 Classes - False Positives ({detection_column})")
        plt.ylabel("Count")
        plt.xlabel("Class")
        plt.xticks(rotation=45)

    # Plot 2: False Negatives by class
    if fn_data:
        fn_df = pd.DataFrame(fn_data)
        fn_class_counts = fn_df["dominant_class"].value_counts().head(10)

        plt.subplot(2, 2, 2)
        fn_class_counts.plot(kind="bar", color="blue", alpha=0.7)
        plt.title(f"Top 10 Classes - False Negatives ({detection_column})")
        plt.ylabel("Count")
        plt.xlabel("Class")
        plt.xticks(rotation=45)

    # Plot 3: Detection Score Distribution for errors
    plt.subplot(2, 2, 3)
    if fp_data:
        fp_scores = [item["detection_score"] for item in fp_data]
        plt.hist(fp_scores, bins=20, alpha=0.7, label="False Positives", color="red")
    if fn_data:
        fn_scores = [item["detection_score"] for item in fn_data]
        plt.hist(fn_scores, bins=20, alpha=0.7, label="False Negatives", color="blue")
    plt.title(f"Detection Score Distribution - Errors ({detection_column})")
    plt.xlabel("Detection Score")
    plt.ylabel("Frequency")
    plt.legend()

    # Plot 4: Error counts comparison
    plt.subplot(2, 2, 4)
    error_counts = [len(fp_data), len(fn_data)]
    error_labels = ["False Positives", "False Negatives"]
    colors = ["red", "blue"]
    plt.bar(error_labels, error_counts, color=colors, alpha=0.7)
    plt.title(f"Error Counts Comparison ({detection_column})")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"error_analysis_detailed_{detection_column}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"✅ Error visualization saved for {detection_column}")


def run_analysis_with_optimal_threshold(
    merged_csv: str, temp_gt_dir: str, detection_column: str, output_dir: str
):
    """
    Analyze with optimal threshold that maximizes F1-score.

    Args:
        merged_csv: Path to the CSV file
        temp_gt_dir: Temporary directory with ground truth
        detection_column: Detection column
        output_dir: Output directory
    """
    try:
        # Add current directory (scripts) to path for relative imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        scripts_dir = os.path.dirname(current_dir)
        sys.path.insert(0, scripts_dir)
        from evaluation.evaluate_detection import (
            load_ground_truth,
            load_predictions,
            create_time_segments,
            analyze_detection_performance,
        )

        print(f"🔍 Searching for optimal threshold for {detection_column}...")

        # Load data with temporary threshold
        ground_truth = load_ground_truth(temp_gt_dir)
        predictions = load_predictions(
            merged_csv, detection_column, threshold=0.5, duration=5.0
        )
        segments_df = create_time_segments(ground_truth, predictions)

        # Find optimal threshold
        optimal_result = find_optimal_threshold(segments_df)
        optimal_threshold = optimal_result["optimal_threshold"]
        optimal_metrics = optimal_result["optimal_metrics"]

        print(f"✅ Optimal threshold found: {optimal_threshold:.3f}")
        print(f"📊 Optimal F1-Score: {optimal_metrics['f1_score']:.3f}")

        # Relaunch analysis with optimal threshold
        optimal_output_dir = os.path.join(output_dir, f"optimal_{detection_column}")
        segments_df_optimal, metrics_optimal = analyze_detection_performance(
            merged_csv,
            temp_gt_dir,
            detection_column,
            optimal_threshold,
            optimal_output_dir,
            duration=5.0,
        )

        # Analyze false positives by class
        fp_analysis = analyze_false_positives_classes(merged_csv, segments_df_optimal)

        # Create detailed error analysis
        error_analysis = create_detailed_error_analysis_table(
            merged_csv, segments_df_optimal, detection_column, optimal_output_dir
        )

        # Create combined error summary
        create_combined_error_summary(
            error_analysis["false_positives_data"],
            error_analysis["false_negatives_data"],
            detection_column,
            optimal_output_dir,
        )

        # Create error visualizations
        create_error_visualization(
            error_analysis["false_positives_data"],
            error_analysis["false_negatives_data"],
            detection_column,
            optimal_output_dir,
        )

        # Save optimal results
        optimal_results = {
            "detection_column": detection_column,
            "optimal_threshold": optimal_threshold,
            "optimal_metrics": optimal_metrics,
            "false_positives_analysis": fp_analysis,
            "total_segments": len(segments_df_optimal),
            "error_analysis": error_analysis,
        }

        # Save as JSON for easy access
        with open(os.path.join(optimal_output_dir, "optimal_analysis.json"), "w") as f:
            json.dump(optimal_results, f, indent=2, default=str)

        return optimal_results

    except Exception as e:
        print(f"❌ Error during optimal analysis of {detection_column}: {e}")
        return None


def run_individual_session_evaluation(
    csv_file: str, annotation_dirs: list, session_name: str
):
    """Run evaluation for a single session."""
    print(f"\n📊 Evaluating individual session: {session_name}")

    # Create session-specific output directory
    session_output = os.path.join(OUTPUT_BASE, f"eval_individual_{session_name}")
    os.makedirs(session_output, exist_ok=True)

    # Create temporary annotations directory for this session
    temp_dir = os.path.join(session_output, "temp_annotations")
    os.makedirs(temp_dir, exist_ok=True)

    # Copy relevant annotation files for this session
    session_parts = session_name.split("_")
    session_date = "_".join(session_parts[:3])  # e.g., "20240408_session_01"

    annotation_files_copied = 0
    for ann_dir in annotation_dirs:
        if session_date in ann_dir["session"]:
            for txt_file in Path(ann_dir["full_path"]).glob("*.txt"):
                dest_path = os.path.join(temp_dir, txt_file.name)
                shutil.copy2(txt_file, dest_path)
                annotation_files_copied += 1

    if annotation_files_copied == 0:
        print(f"⚠️  No annotations found for session {session_name}")
        shutil.rmtree(temp_dir)
        return

    print(f"📝 {annotation_files_copied} annotation files found for {session_name}")

    try:
        # Add the scripts directory to path and import
        scripts_dir = os.path.dirname(current_dir)
        sys.path.insert(0, scripts_dir)
        from evaluation.evaluate_detection import analyze_detection_performance

        # Columns to evaluate
        columns = ["tag_Buzz", "tag_Insect", "tag_Bird", "buzz", "biophony"]

        session_results = []
        session_optimal_results = []

        for col in columns:
            print(f"  🔍 Evaluating {col} for {session_name}")

            eval_dir = os.path.join(session_output, f"eval_{col}")

            try:
                # Standard evaluation
                segments_df, metrics = analyze_detection_performance(
                    csv_file, temp_dir, col, 0.5, eval_dir, duration=5.0
                )

                session_results.append(
                    {
                        "session": session_name,
                        "column": col,
                        "f1_score": metrics["f1_score"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "accuracy": metrics["accuracy"],
                    }
                )

                # Optimal threshold evaluation
                optimal_result = run_analysis_with_optimal_threshold(
                    csv_file, temp_dir, col, eval_dir
                )

                if optimal_result:
                    session_optimal_results.append(
                        {
                            "session": session_name,
                            "column": col,
                            "default_f1": metrics["f1_score"],
                            "default_precision": metrics["precision"],
                            "default_recall": metrics["recall"],
                            "optimal_threshold": optimal_result["optimal_threshold"],
                            "optimal_f1": optimal_result["optimal_metrics"]["f1_score"],
                            "optimal_precision": optimal_result["optimal_metrics"][
                                "precision"
                            ],
                            "optimal_recall": optimal_result["optimal_metrics"][
                                "recall"
                            ],
                            "improvement": optimal_result["optimal_metrics"]["f1_score"]
                            - metrics["f1_score"],
                            "false_positives": optimal_result["error_analysis"][
                                "false_positives_count"
                            ],
                            "false_negatives": optimal_result["error_analysis"][
                                "false_negatives_count"
                            ],
                        }
                    )

            except Exception as e:
                print(f"    ❌ Error evaluating {col}: {e}")

        # Save session results
        if session_results:
            session_df = pd.DataFrame(session_results)
            session_summary_path = os.path.join(
                session_output, f"evaluation_summary_{session_name}.csv"
            )
            session_df.to_csv(session_summary_path, index=False)

            print(f"\n📊 SESSION RESULTS - {session_name}:")
            print(
                session_df[["column", "f1_score", "precision", "recall"]].to_string(
                    index=False, float_format="%.3f"
                )
            )

        if session_optimal_results:
            session_optimal_df = pd.DataFrame(session_optimal_results)
            session_optimal_path = os.path.join(
                session_output, f"optimal_evaluation_summary_{session_name}.csv"
            )
            session_optimal_df.to_csv(session_optimal_path, index=False)

            # Create session-specific comparison plot
            create_threshold_comparison_plot(session_optimal_df, session_output)

        # Cleanup
        shutil.rmtree(temp_dir)

        print(f"✅ Session {session_name} evaluation completed")
        return session_results, session_optimal_results

    except Exception as e:
        print(f"❌ Error during session evaluation: {e}")
        shutil.rmtree(temp_dir)
        return None, None


def run_combined_evaluation(merged_csv: str, annotation_dirs: list):
    """Run evaluation on all combined data."""
    print(f"\n📊 Running combined evaluation on all sessions")

    # Create combined annotations
    combined_annotations = create_combined_annotations(annotation_dirs)

    try:
        run_evaluation(merged_csv, combined_annotations)
        # Cleanup
        shutil.rmtree(combined_annotations)
        print(f"✅ Combined evaluation completed")

    except Exception as e:
        print(f"❌ Error during combined evaluation: {e}")
        if os.path.exists(combined_annotations):
            shutil.rmtree(combined_annotations)


def create_cross_session_comparison(
    all_session_results: list, all_optimal_results: list
):
    """Create comparison plots across all sessions."""
    if not all_session_results:
        return

    print(f"\n📈 Creating cross-session comparison")

    # Combine all session results
    all_results_df = pd.DataFrame(
        [
            result
            for session_results in all_session_results
            for result in session_results
        ]
    )
    all_optimal_df = pd.DataFrame(
        [
            result
            for optimal_results in all_optimal_results
            for result in optimal_results
        ]
    )

    # Remove duplicates if any (keep first occurrence)
    print(f"📊 Standard results before dedup: {len(all_results_df)} rows")
    all_results_df = all_results_df.drop_duplicates(
        subset=["session", "column"], keep="first"
    )
    print(f"📊 Standard results after dedup: {len(all_results_df)} rows")

    if not all_optimal_df.empty:
        print(f"📊 Optimal results before dedup: {len(all_optimal_df)} rows")
        all_optimal_df = all_optimal_df.drop_duplicates(
            subset=["session", "column"], keep="first"
        )
        print(f"📊 Optimal results after dedup: {len(all_optimal_df)} rows")

    # Save combined session results
    cross_session_dir = os.path.join(OUTPUT_BASE, "cross_session_analysis")
    os.makedirs(cross_session_dir, exist_ok=True)

    all_results_df.to_csv(
        os.path.join(cross_session_dir, "all_sessions_standard.csv"), index=False
    )
    if not all_optimal_df.empty:
        all_optimal_df.to_csv(
            os.path.join(cross_session_dir, "all_sessions_optimal.csv"), index=False
        )

    # Create visualization comparing sessions
    plt.figure(figsize=(20, 12))

    # Check if we have enough data for pivot
    if len(all_results_df) == 0:
        print("⚠️  No data available for cross-session comparison")
        return

    try:
        # F1-Score comparison by session and column
        pivot_f1 = all_results_df.pivot(
            index="session", columns="column", values="f1_score"
        )
    except ValueError as e:
        print(f"⚠️  Cannot create pivot table: {e}")
        print("📊 Creating alternative visualization...")

        # Alternative: grouped bar chart
        plt.subplot(2, 1, 1)
        for column in all_results_df["column"].unique():
            column_data = all_results_df[all_results_df["column"] == column]
            plt.plot(
                column_data["session"],
                column_data["f1_score"],
                marker="o",
                label=column,
                linewidth=2,
            )

        plt.title("F1-Score by Session and Column (Standard)")
        plt.ylabel("F1-Score")
        plt.xlabel("Session")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(cross_session_dir, "cross_session_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Create summary statistics
        summary_stats = (
            all_results_df.groupby("column")
            .agg(
                {
                    "f1_score": ["mean", "std", "min", "max"],
                    "precision": ["mean", "std"],
                    "recall": ["mean", "std"],
                }
            )
            .round(4)
        )

        summary_stats.to_csv(
            os.path.join(cross_session_dir, "column_performance_summary.csv")
        )

        print(f"✅ Cross-session analysis saved in {cross_session_dir}")

        plt.subplot(2, 2, 1)
        sns.heatmap(
            pivot_f1,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            cbar_kws={"label": "F1-Score"},
        )
        plt.title("F1-Score by Session and Column (Standard)")
        plt.ylabel("Session")
        plt.xlabel("Column")

        # Optimal F1-Score comparison
        if not all_optimal_df.empty:
            try:
                pivot_optimal_f1 = all_optimal_df.pivot(
                    index="session", columns="column", values="optimal_f1"
                )

                plt.subplot(2, 2, 2)
                sns.heatmap(
                    pivot_optimal_f1,
                    annot=True,
                    fmt=".3f",
                    cmap="Greens",
                    cbar_kws={"label": "Optimal F1-Score"},
                )
                plt.title("F1-Score by Session and Column (Optimal)")
                plt.ylabel("Session")
                plt.xlabel("Column")

                # Improvement comparison
                pivot_improvement = all_optimal_df.pivot(
                    index="session", columns="column", values="improvement"
                )

                plt.subplot(2, 2, 3)
                sns.heatmap(
                    pivot_improvement,
                    annot=True,
                    fmt=".3f",
                    cmap="RdYlGn",
                    center=0,
                    cbar_kws={"label": "F1-Score Improvement"},
                )
                plt.title("F1-Score Improvement by Session and Column")
                plt.ylabel("Session")
                plt.xlabel("Column")

            except ValueError as e:
                print(f"⚠️  Cannot create optimal pivot tables: {e}")
                # Create alternative plot for optimal results
                plt.subplot(2, 2, 2)
                for column in all_optimal_df["column"].unique():
                    column_data = all_optimal_df[all_optimal_df["column"] == column]
                    plt.plot(
                        column_data["session"],
                        column_data["optimal_f1"],
                        marker="s",
                        label=column,
                        linewidth=2,
                    )

                plt.title("Optimal F1-Score by Session and Column")
                plt.ylabel("Optimal F1-Score")
                plt.xlabel("Session")
                plt.legend()
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)

        # Average performance by column
        avg_by_column = (
            all_results_df.groupby("column")["f1_score"]
            .mean()
            .sort_values(ascending=False)
        )

        plt.subplot(2, 2, 4)
        avg_by_column.plot(kind="bar", color="skyblue")
        plt.title("Average F1-Score by Column Across All Sessions")
        plt.ylabel("Average F1-Score")
        plt.xlabel("Column")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(
            os.path.join(cross_session_dir, "cross_session_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Create summary statistics
        summary_stats = (
            all_results_df.groupby("column")
            .agg(
                {
                    "f1_score": ["mean", "std", "min", "max"],
                    "precision": ["mean", "std"],
                    "recall": ["mean", "std"],
                }
            )
            .round(4)
        )

        summary_stats.to_csv(
            os.path.join(cross_session_dir, "column_performance_summary.csv")
        )

        print(f"✅ Cross-session analysis saved in {cross_session_dir}")


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Automated processing with advanced evaluation"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML format)",
    )
    parser.add_argument(
        "--analysis-mode",
        choices=["individual", "combined", "both"],
        help="Analysis mode: individual sessions, combined, or both",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        help="Specific sessions to analyze (e.g., 20240408_session_01_Tent_SM05_T)",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        help="Columns to evaluate",
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip data processing and go directly to evaluation",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print current configuration and exit",
    )

    args = parser.parse_args()

    # Load configuration with potential override
    global config_data
    if args.config:
        try:
            from config.config_loader import WorkflowConfig

            config_data = WorkflowConfig(args.config)
        except ImportError:
            print(f"⚠️  Could not load custom config file: {args.config}")
            print("📝 Using default configuration")

    # Print configuration if requested
    if args.print_config:
        if config_data:
            config_data.print_config()
        else:
            print("📋 DEFAULT CONFIGURATION:")
            print("=" * 50)
            print(f"📁 Data path: {BASE_DATA_PATH}")
            print(f"🐍 Python path: {PYTHON_PATH}")
            print(f"📜 Process script: {PROCESS_SCRIPT}")
            print(f"📤 Output base: {OUTPUT_BASE}")
            print(f"⏱️  Segment length: {SEGMENT_LENGTH}s")
        return

    # Get configuration values (command line overrides config file)
    analysis_mode = args.analysis_mode or (config.analysis_mode if config else "both")
    target_sessions = args.sessions or (
        config.target_sessions if config and hasattr(config, "target_sessions") else []
    )
    columns = args.columns or (
        config.columns if config else ["tag_Buzz", "tag_Insect", "biophony"]
    )
    skip_processing = args.skip_processing or (
        config.skip_processing
        if config and hasattr(config, "skip_processing")
        else False
    )

    print("🚀 COMPLETE AUTOMATED PROCESSING")
    print("=" * 50)
    if config:
        print(f"⚙️  Using config: {config.config_file}")
    print(f"📊 Analysis mode: {analysis_mode}")
    if target_sessions:
        print(f"🎯 Target sessions: {', '.join(target_sessions)}")
    print(f"📈 Columns to evaluate: {', '.join(columns)}")
    if skip_processing:
        print(f"⏩ Skipping data processing")

    # Create output directory
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Step 1: Directory discovery
    print("\n1️⃣ Directory discovery...")
    data_dirs = find_data_directories(BASE_DATA_PATH)
    annotation_dirs = find_annotation_directories(BASE_DATA_PATH)

    print(f"📁 {len(data_dirs)} data directories found")
    print(f"📝 {len(annotation_dirs)} annotation directories found")

    if not data_dirs or not annotation_dirs:
        print("❌ Insufficient data!")
        return

    # Filter sessions if specified
    if target_sessions:
        filtered_data_dirs = []
        for data_dir in data_dirs:
            session_name = f"{data_dir['session']}_{data_dir['subdirectory']}"
            if session_name in target_sessions:
                filtered_data_dirs.append(data_dir)
        data_dirs = filtered_data_dirs
        print(f"🎯 Filtered to {len(data_dirs)} target sessions")

    if not skip_processing:
        # Check existing processing status
        print("\n🔍 Checking existing processing status...")
        existing_csv_files = []
        new_processing_needed = []

        for data_dir in data_dirs:
            session_name = f"{data_dir['session']}_{data_dir['subdirectory']}"
            output_dir = os.path.join(OUTPUT_BASE, session_name)
            csv_file = os.path.join(output_dir, f"indices_{session_name}.csv")

            if os.path.exists(csv_file):
                existing_csv_files.append(csv_file)
            else:
                new_processing_needed.append(data_dir)

        print(f"✅ {len(existing_csv_files)} directories already processed")
        print(f"🔄 {len(new_processing_needed)} directories need processing")

        # Check merged file
        merged_path = os.path.join(OUTPUT_BASE, "merged_results.csv")
        if os.path.exists(merged_path):
            print(f"📊 Merged results file already exists: {merged_path}")
        else:
            print(f"📊 Merged results file will be created")

        # Step 2: Data processing
        print("\n2️⃣ Data processing...")
        csv_files = existing_csv_files.copy()  # Start with existing files

        for i, data_dir in enumerate(data_dirs, 1):
            print(f"[{i}/{len(data_dirs)}] ", end="")
            csv_file = run_process_for_directory(data_dir)
            if csv_file:
                csv_files.append(csv_file)

        # Step 3: Results merging
        print(f"\n3️⃣ Results merging...")
        if not csv_files:
            print("❌ No CSV files generated")
            return

        merged_csv = merge_csv_files(csv_files)
        if not merged_csv:
            print("❌ Merge failed")
            return
    else:
        print("\n⏩ Skipping data processing...")
        # Find existing CSV files
        csv_files = []
        for data_dir in data_dirs:
            session_name = f"{data_dir['session']}_{data_dir['subdirectory']}"
            output_dir = os.path.join(OUTPUT_BASE, session_name)
            csv_file = os.path.join(output_dir, f"indices_{session_name}.csv")
            if os.path.exists(csv_file):
                csv_files.append(csv_file)

        merged_csv = os.path.join(OUTPUT_BASE, "merged_results.csv")
        if not os.path.exists(merged_csv):
            print("🔄 Creating merged file...")
            merged_csv = merge_csv_files(csv_files)
            if not merged_csv:
                print("❌ Failed to create merged file")
                return

    # Step 4: Evaluation based on analysis mode
    print("\n4️⃣ Evaluation...")

    all_session_results = []
    all_optimal_results = []

    if analysis_mode in ["individual", "both"]:
        print("\n📊 Individual session analysis...")
        for csv_file in csv_files:
            # Extract session name from CSV file path
            session_name = os.path.basename(os.path.dirname(csv_file))

            session_results, session_optimal = run_individual_session_evaluation(
                csv_file, annotation_dirs, session_name
            )

            if session_results:
                all_session_results.append(session_results)
            if session_optimal:
                all_optimal_results.append(session_optimal)

    if analysis_mode in ["combined", "both"]:
        print("\n📊 Combined analysis...")
        run_combined_evaluation(merged_csv, annotation_dirs)

    # Step 5: Cross-session comparison (if individual analysis was done)
    if analysis_mode in ["individual", "both"] and all_session_results:
        print("\n5️⃣ Cross-session comparison...")
        create_cross_session_comparison(all_session_results, all_optimal_results)

    print(f"\n✅ COMPLETED!")
    print(f"📁 Results in: {OUTPUT_BASE}")
    print(f"\n📊 ANALYSIS RESULTS:")

    if analysis_mode in ["individual", "both"]:
        print(
            f"  📈 Individual session results: {OUTPUT_BASE}/eval_individual_[session]/"
        )
        print(f"  📊 Cross-session comparison: {OUTPUT_BASE}/cross_session_analysis/")

    if analysis_mode in ["combined", "both"]:
        print(f"  🔗 Combined analysis: {OUTPUT_BASE}/eval_[column]/")
        print(
            f"  📋 Summary files: evaluation_summary.csv, optimal_evaluation_summary.csv"
        )

    print(f"\n🎯 ADVANCED FEATURES INCLUDED:")
    print(f"  🎯 Optimal threshold analysis for each column")
    print(f"  📈 Detailed false positive/negative analysis")
    print(f"  📊 Class-based error breakdown")
    print(f"  📉 Performance comparison visualizations")
    print(f"  💾 Comprehensive CSV reports and JSON summaries")

    if analysis_mode in ["individual", "both"]:
        print(f"  🔍 Session-by-session performance analysis")
        print(f"  📈 Cross-session performance comparison")


if __name__ == "__main__":
    main()
