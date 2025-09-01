#!/usr/bin/env python3
"""
Simplified audio detection batch processing script.
Processes all audio directories found in the data path and generates detection results.
Supports both inference mode (detection only) and evaluation mode (detection + evaluation).
"""

import os
import sys
import subprocess
import pandas as pd
import glob
import argparse
from pathlib import Path
import shutil

# Add config to path
current_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(current_dir, "scripts", "config")
sys.path.append(config_dir)

try:
    from scripts.config.config_minimal import WorkflowConfig
except ImportError:
    print("Error: Cannot load configuration system")
    sys.exit(1)


def find_data_directories(base_data_path):
    """Find all valid data directories containing audio files."""
    if not os.path.exists(base_data_path):
        print(f"Error: Data path {base_data_path} does not exist")
        return []

    directories = []
    for item in os.listdir(base_data_path):
        item_path = os.path.join(base_data_path, item)
        if os.path.isdir(item_path) and not item.endswith("_annotées"):
            # Look for subdirectories with audio files
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path) and not subitem.endswith("_annotées"):
                    # Count audio files
                    audio_files = glob.glob(os.path.join(subitem_path, "*.wav"))
                    audio_files.extend(glob.glob(os.path.join(subitem_path, "*.flac")))

                    if audio_files:
                        session_name = f"{item}_{subitem}"
                        directories.append(
                            {
                                "session": item,
                                "subdirectory": subitem,
                                "name": session_name,
                                "path": subitem_path,
                                "audio_count": len(audio_files),
                            }
                        )

    return directories


def find_annotation_directories(base_data_path, annotation_pattern):
    """Find all annotation directories."""
    if not os.path.exists(base_data_path):
        return []

    annotation_dirs = []
    for item in os.listdir(base_data_path):
        item_path = os.path.join(base_data_path, item)
        if os.path.isdir(item_path):
            for subitem in os.listdir(item_path):
                if subitem.endswith("_annotées"):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path):
                        annotation_files = glob.glob(
                            os.path.join(subitem_path, "*.txt")
                        )
                        if annotation_files:
                            annotation_dirs.append(
                                {
                                    "session": item,
                                    "subdirectory": subitem,
                                    "path": subitem_path,
                                    "annotation_count": len(annotation_files),
                                }
                            )

    return annotation_dirs


def run_site_level_evaluation(result_file, session_name, annotation_dirs, config):
    """Run evaluation for a single site/session."""
    if not config.is_evaluation_mode:
        return

    # Check evaluation scope
    scope = getattr(config, "evaluation_scope", "both")
    if scope == "global":
        return  # Skip site-level evaluation if only global is requested

    print(f"Running site-level evaluation for: {session_name}")

    # Create site-specific output directories
    session_output_dir = os.path.join(config.output_base, session_name)
    classical_eval_dir = os.path.join(session_output_dir, "classical_evaluation")
    advanced_eval_dir = os.path.join(session_output_dir, "advanced_evaluation")

    os.makedirs(classical_eval_dir, exist_ok=True)
    os.makedirs(advanced_eval_dir, exist_ok=True)

    # Find relevant annotation directory for this session
    session_annotations = []
    session_prefix = session_name.split("_")[0]  # Extract session date

    for ann_dir in annotation_dirs:
        if session_prefix in ann_dir["session"]:
            session_annotations.append(ann_dir)

    if not session_annotations:
        print(f"  No annotations found for session {session_name}")
        return

    # Create temporary annotations directory for this session
    temp_annotations = os.path.join(classical_eval_dir, "temp_annotations")
    os.makedirs(temp_annotations, exist_ok=True)

    # Copy relevant annotation files
    for ann_dir in session_annotations:
        for txt_file in Path(ann_dir["path"]).glob("*.txt"):
            dest_path = os.path.join(temp_annotations, txt_file.name)
            shutil.copy2(txt_file, dest_path)

    try:
        # Add evaluation scripts to path
        eval_dir = os.path.join(current_dir, "scripts", "evaluation")
        sys.path.append(eval_dir)
        from evaluate_detection import (
            analyze_detection_performance,
            run_advanced_evaluation as run_adv_eval,
        )

        # Run classical evaluation for each column
        print(f"  Running classical evaluation...")
        for column in config.columns:
            eval_output_dir = os.path.join(classical_eval_dir, f"eval_{column}")
            os.makedirs(eval_output_dir, exist_ok=True)

            try:
                segments_df, metrics = analyze_detection_performance(
                    result_file,
                    temp_annotations,
                    column,
                    config.default_threshold,
                    eval_output_dir,
                    duration=config.evaluation_duration,
                )

                print(
                    f"    {column}: F1={metrics['f1_score']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}"
                )

            except Exception as e:
                print(f"    Error evaluating {column}: {e}")

        # Run advanced evaluation if enabled
        if hasattr(config, "advanced_evaluation") and config.advanced_evaluation.get(
            "enabled", False
        ):
            print(f"  Running advanced evaluation...")

            # Get advanced evaluation parameters from config
            optimize_threshold = config.advanced_evaluation.get(
                "optimize_threshold", True
            )
            duration = config.advanced_evaluation.get("duration", None)
            excluded_classes = config.advanced_evaluation.get("excluded_classes", {})

            for column in config.columns:
                column_output_dir = os.path.join(
                    advanced_eval_dir, f"advanced_{column}"
                )
                os.makedirs(column_output_dir, exist_ok=True)

                try:
                    results = run_adv_eval(
                        csv_path=result_file,
                        annotations_dir=temp_annotations,
                        detection_column=column,
                        output_dir=column_output_dir,
                        duration=duration,
                        optimize_threshold=optimize_threshold,
                        excluded_classes=excluded_classes,
                    )

                    print(f"    Advanced evaluation completed for {column}")

                    if results and "metrics" in results:
                        metrics = results["metrics"]
                        print(
                            f"      F1={metrics.get('f1_score', 0):.3f}, "
                            f"Precision={metrics.get('precision', 0):.3f}, "
                            f"Recall={metrics.get('recall', 0):.3f}"
                        )

                except Exception as e:
                    print(f"    Error in advanced evaluation for {column}: {e}")

    except ImportError:
        print("  Error: Evaluation scripts not available")
    except Exception as e:
        print(f"  Error during site evaluation: {e}")
    finally:
        # Cleanup
        if os.path.exists(temp_annotations):
            shutil.rmtree(temp_annotations)


def run_global_evaluation(merged_csv, annotation_dirs, config):
    """Run global evaluation on all merged results (renamed from run_evaluation)."""
    if not config.is_evaluation_mode:
        print("Skipping evaluation (inference mode)")
        return

    if not annotation_dirs:
        print("No annotation directories found for evaluation")
        return

    print("Running global classical evaluation")

    # Create global results directory (renamed)
    results_dir = os.path.join(config.output_base, "classical_results")
    os.makedirs(results_dir, exist_ok=True)

    # Create combined annotations directory
    temp_annotations = os.path.join(results_dir, "temp_annotations")
    os.makedirs(temp_annotations, exist_ok=True)

    # Copy all annotation files
    for ann_dir in annotation_dirs:
        for txt_file in Path(ann_dir["path"]).glob("*.txt"):
            dest_path = os.path.join(temp_annotations, txt_file.name)
            shutil.copy2(txt_file, dest_path)

    try:
        # Add evaluation scripts to path
        eval_dir = os.path.join(current_dir, "scripts", "evaluation")
        sys.path.append(eval_dir)
        from evaluate_detection import analyze_detection_performance

        # Run evaluation for each column
        for column in config.columns:
            print(f"Evaluating column: {column}")

            eval_output_dir = os.path.join(results_dir, f"eval_{column}")
            os.makedirs(eval_output_dir, exist_ok=True)

            try:
                segments_df, metrics = analyze_detection_performance(
                    merged_csv,
                    temp_annotations,
                    column,
                    config.default_threshold,
                    eval_output_dir,
                    duration=config.evaluation_duration,
                )

                print(
                    f"  {column}: F1={metrics['f1_score']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}"
                )

            except Exception as e:
                print(f"  Error evaluating {column}: {e}")

        print(f"Global classical evaluation results saved in: {results_dir}")

    except ImportError:
        print("Error: Evaluation scripts not available")
    except Exception as e:
        print(f"Error during evaluation: {e}")
    finally:
        # Cleanup
        if os.path.exists(temp_annotations):
            shutil.rmtree(temp_annotations)


def run_global_advanced_evaluation(merged_csv, annotation_dirs, config):
    """Run global advanced evaluation tasks with parameters from config (renamed from run_advanced_evaluation)."""
    print("Running global advanced evaluation tasks")

    try:
        # Add evaluation scripts to path
        eval_dir = os.path.join(current_dir, "scripts", "evaluation")
        sys.path.append(eval_dir)
        from evaluate_detection import run_advanced_evaluation as run_adv_eval

        # Set up global advanced evaluation output directory (renamed)
        if hasattr(config, "advanced_evaluation") and config.advanced_evaluation.get(
            "output_dir"
        ):
            advanced_output_dir = config.advanced_evaluation["output_dir"].replace(
                "advanced_evaluation", "global_advanced_evaluation"
            )
        else:
            advanced_output_dir = os.path.join(
                config.output_base, "global_advanced_evaluation"
            )

        os.makedirs(advanced_output_dir, exist_ok=True)

        # Create combined annotations directory
        temp_annotations = os.path.join(advanced_output_dir, "temp_annotations")
        os.makedirs(temp_annotations, exist_ok=True)

        # Copy all annotation files
        for ann_dir in annotation_dirs:
            for txt_file in Path(ann_dir["path"]).glob("*.txt"):
                dest_path = os.path.join(temp_annotations, txt_file.name)
                shutil.copy2(txt_file, dest_path)

        # Get advanced evaluation parameters from config
        optimize_threshold = True
        duration = None
        excluded_classes = {}

        if hasattr(config, "advanced_evaluation"):
            optimize_threshold = config.advanced_evaluation.get(
                "optimize_threshold", True
            )
            duration = config.advanced_evaluation.get("duration", None)
            excluded_classes = config.advanced_evaluation.get("excluded_classes", {})

        # Run advanced evaluation for each column
        for column in config.columns:
            print(f"Running global advanced evaluation for column: {column}")

            column_output_dir = os.path.join(advanced_output_dir, f"advanced_{column}")
            os.makedirs(column_output_dir, exist_ok=True)

            try:
                # Call advanced evaluation with config parameters
                results = run_adv_eval(
                    csv_path=merged_csv,
                    annotations_dir=temp_annotations,
                    detection_column=column,
                    output_dir=column_output_dir,
                    duration=duration,
                    optimize_threshold=optimize_threshold,
                    excluded_classes=excluded_classes,
                )

                print(f"  Global advanced evaluation completed for {column}")

                # Print summary metrics
                if results and "metrics" in results:
                    metrics = results["metrics"]
                    print(
                        f"    F1={metrics.get('f1_score', 0):.3f}, "
                        f"Precision={metrics.get('precision', 0):.3f}, "
                        f"Recall={metrics.get('recall', 0):.3f}"
                    )

            except Exception as e:
                print(f"  Error in global advanced evaluation for {column}: {e}")

        print(f"Global advanced evaluation results saved in: {advanced_output_dir}")

        # Cleanup
        if os.path.exists(temp_annotations):
            shutil.rmtree(temp_annotations)

    except ImportError:
        print("Error: Advanced evaluation scripts not available")
    except Exception as e:
        print(f"Error during global advanced evaluation: {e}")


def run_detection_for_directory(data_dir, config, annotation_dirs=None):
    """Run detection processing for a single directory and site-level evaluation."""
    session_name = data_dir["name"]
    data_path = data_dir["path"]

    # Create output directory
    output_dir = os.path.join(config.output_base, session_name)
    os.makedirs(output_dir, exist_ok=True)

    # Check if already processed
    result_file = os.path.join(output_dir, f"indices_{session_name}.csv")
    if os.path.exists(result_file):
        print(f"Already processed: {session_name}")
    else:
        print(f"Processing: {session_name} ({data_dir['audio_count']} audio files)")

        # Run detection process
        cmd = [
            config.python_path,
            config.process_script,
            "--data_path",
            data_path,
            "--save_path",
            output_dir,
            "--name",
            session_name,
            "--audio_format",
            config.audio_format,
            "--l",
            str(config.segment_length),
            "--model",
            config.model_type,
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=config.timeout
            )

            if result.returncode == 0:
                if os.path.exists(result_file):
                    print(f"Success: {session_name}")
                else:
                    print(f"Error: Result file not found for {session_name}")
                    return None
            else:
                print(f"Error processing {session_name}: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print(f"Timeout processing {session_name}")
            return None
        except Exception as e:
            print(f"Exception processing {session_name}: {e}")
            return None

    # Run site-level evaluation if we have the result file and annotations
    if os.path.exists(result_file) and annotation_dirs and config.is_evaluation_mode:
        run_site_level_evaluation(result_file, session_name, annotation_dirs, config)

    return result_file if os.path.exists(result_file) else None


def merge_results(csv_files, output_path):
    """Merge all CSV result files."""
    if not csv_files:
        print("No CSV files to merge")
        return None

    merged_file = os.path.join(output_path, "merged_results.csv")

    # Check if already exists and is up to date
    if os.path.exists(merged_file):
        try:
            existing_df = pd.read_csv(merged_file)
            total_expected_rows = 0
            for csv_file in csv_files:
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    total_expected_rows += len(df)

            if len(existing_df) >= total_expected_rows * 0.95:  # 5% tolerance
                print(f"Merged file already exists and is up to date: {merged_file}")
                return merged_file
        except Exception as e:
            print(f"Error checking existing merged file: {e}")

    print(f"Merging {len(csv_files)} result files")

    all_dfs = []
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                all_dfs.append(df)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.to_csv(merged_file, index=False)
        print(f"Merged results saved: {merged_file} ({len(merged_df)} rows)")
        return merged_file

    return None


def run_evaluation(merged_csv, annotation_dirs, config):
    """Run evaluation on the merged results."""
    if not config.is_evaluation_mode:
        print("Skipping evaluation (inference mode)")
        return

    if not annotation_dirs:
        print("No annotation directories found for evaluation")
        return

    print("Running standard evaluation")

    # Create results directory
    results_dir = os.path.join(config.output_base, config.results_subdir)
    os.makedirs(results_dir, exist_ok=True)

    # Create combined annotations directory
    temp_annotations = os.path.join(results_dir, "temp_annotations")
    os.makedirs(temp_annotations, exist_ok=True)

    # Copy all annotation files
    for ann_dir in annotation_dirs:
        for txt_file in Path(ann_dir["path"]).glob("*.txt"):
            dest_path = os.path.join(temp_annotations, txt_file.name)
            shutil.copy2(txt_file, dest_path)

    try:
        # Add evaluation scripts to path
        eval_dir = os.path.join(current_dir, "scripts", "evaluation")
        sys.path.append(eval_dir)
        from evaluate_detection import analyze_detection_performance

        # Run evaluation for each column
        for column in config.columns:
            print(f"Evaluating column: {column}")

            eval_output_dir = os.path.join(results_dir, f"eval_{column}")
            os.makedirs(eval_output_dir, exist_ok=True)

            try:
                segments_df, metrics = analyze_detection_performance(
                    merged_csv,
                    temp_annotations,
                    column,
                    config.default_threshold,
                    eval_output_dir,
                    duration=config.evaluation_duration,
                )

                print(
                    f"  {column}: F1={metrics['f1_score']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}"
                )

            except Exception as e:
                print(f"  Error evaluating {column}: {e}")

        print(f"Standard evaluation results saved in: {results_dir}")

    except ImportError:
        print("Error: Evaluation scripts not available")
    except Exception as e:
        print(f"Error during evaluation: {e}")
    finally:
        # Cleanup
        if os.path.exists(temp_annotations):
            shutil.rmtree(temp_annotations)


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description="Simplified audio detection batch processing"
    )
    parser.add_argument(
        "--config",
        default="scripts/config/simple_config.yaml",
        help="Configuration file path",
    )
    parser.add_argument("--sessions", nargs="*", help="Specific sessions to process")
    parser.add_argument(
        "--mode",
        choices=["inference", "evaluation"],
        help="Override analysis mode from config",
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip detection processing, go directly to evaluation",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = WorkflowConfig(args.config)
        print(f"Configuration loaded: {args.config}")
        print(f"Analysis mode: {config.analysis_mode}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Override mode if specified
    if args.mode:
        config.set_mode(args.mode)
        print(f"Mode overridden to: {args.mode}")

    # Create output directory
    os.makedirs(config.output_base, exist_ok=True)

    # Find annotation directories (needed for both site-level and global evaluation)
    annotation_dirs = []
    if config.is_evaluation_mode:
        annotation_dirs = find_annotation_directories(
            config.data_path, config.annotation_pattern
        )

    if not args.skip_processing:
        # Find data directories
        data_dirs = find_data_directories(config.data_path)
        if not data_dirs:
            print(f"No valid data directories found in {config.data_path}")
            return 1

        # Filter by sessions if specified
        if args.sessions:
            data_dirs = [d for d in data_dirs if d["name"] in args.sessions]
            if not data_dirs:
                print("No matching sessions found")
                return 1

        print(f"Found {len(data_dirs)} directories to process")

        # Process each directory with site-level evaluation
        result_files = []
        for data_dir in data_dirs:
            result_file = run_detection_for_directory(data_dir, config, annotation_dirs)
            if result_file:
                result_files.append(result_file)

        # Merge results
        if result_files:
            merged_file = merge_results(result_files, config.output_base)
        else:
            print("No successful processing results")
            return 1
    else:
        # Look for existing merged file
        merged_file = os.path.join(config.output_base, "merged_results.csv")
        if not os.path.exists(merged_file):
            print(f"No merged results file found at {merged_file}")
            return 1
        print(f"Using existing merged results: {merged_file}")

    # Run evaluation based on evaluation_scope configuration
    if config.is_evaluation_mode:
        scope = getattr(config, "evaluation_scope", "both")

        if scope in ["global", "both"]:
            print(f"Running global evaluation (scope: {scope})")
            # Run global classical evaluation
            run_global_evaluation(merged_file, annotation_dirs, config)

            # Run global advanced evaluation if enabled in config
            if hasattr(
                config, "advanced_evaluation"
            ) and config.advanced_evaluation.get("enabled", False):
                print("Global advanced evaluation is enabled in config")
                run_global_advanced_evaluation(merged_file, annotation_dirs, config)
            else:
                print("Global advanced evaluation is disabled in config")

        if scope == "local":
            print("Only local (site-level) evaluation was performed")
        elif scope == "global":
            print("Only global evaluation was performed")

    print(f"Processing complete. Results in: {config.output_base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
