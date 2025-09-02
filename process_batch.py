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
import json
import signal
from datetime import datetime

# Add config to path
current_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(current_dir, "scripts", "config")
sys.path.append(config_dir)

try:
    from scripts.config.config_minimal import WorkflowConfig
except ImportError:
    print("Error: Cannot load configuration system")
    sys.exit(1)


def get_evaluation_functions():
    """Dynamically import evaluation functions."""
    try:
        eval_dir = os.path.join(current_dir, "scripts", "evaluation")
        sys.path.insert(0, eval_dir)  # Insert at beginning of path
        import importlib

        evaluate_detection = importlib.import_module("evaluate_detection")
        return (
            evaluate_detection.analyze_detection_performance,
            evaluate_detection.run_advanced_evaluation,
        )
    except ImportError as e:
        print(f"Warning: Could not import evaluation functions: {e}")
        return None, None


# Global variables for checkpoint management
checkpoint_file = None
processed_sessions = set()
total_sessions = 0
current_session_idx = 0


def save_checkpoint(output_base, processed_sessions, current_idx, total_sessions):
    """Save processing state to checkpoint file."""
    checkpoint_data = {
        "timestamp": datetime.now().isoformat(),
        "processed_sessions": list(processed_sessions),
        "current_session_idx": current_idx,
        "total_sessions": total_sessions,
        "completion_percentage": (
            (len(processed_sessions) / total_sessions * 100)
            if total_sessions > 0
            else 0
        ),
    }

    checkpoint_path = os.path.join(output_base, "processing_checkpoint.json")
    try:
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        print(
            f"✓ Checkpoint saved: {len(processed_sessions)}/{total_sessions} sessions processed"
        )
    except Exception as e:
        print(f"⚠️ Warning: Could not save checkpoint: {e}")


def load_checkpoint(output_base):
    """Load processing state from checkpoint file."""
    checkpoint_path = os.path.join(output_base, "processing_checkpoint.json")

    if not os.path.exists(checkpoint_path):
        return {
            "processed_sessions": [],
            "current_session_idx": 0,
            "total_sessions": 0,
            "completion_percentage": 0,
            "timestamp": None,
        }

    try:
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        processed_sessions = set(checkpoint_data.get("processed_sessions", []))
        current_idx = checkpoint_data.get("current_session_idx", 0)
        total_sessions = checkpoint_data.get("total_sessions", 0)

        completion = checkpoint_data.get("completion_percentage", 0)
        timestamp = checkpoint_data.get("timestamp", "unknown")

        print(
            f"✓ Checkpoint loaded: {len(processed_sessions)}/{total_sessions} sessions already processed"
        )
        print(f"  Last saved: {timestamp} ({completion:.1f}% complete)")

        return {
            "processed_sessions": list(processed_sessions),
            "current_session_idx": current_idx,
            "total_sessions": total_sessions,
            "completion_percentage": completion,
            "timestamp": timestamp,
        }

    except Exception as e:
        print(f"⚠️ Warning: Could not load checkpoint: {e}")
        return {
            "processed_sessions": [],
            "current_session_idx": 0,
            "total_sessions": 0,
            "completion_percentage": 0,
            "timestamp": None,
        }


def signal_handler(signum, frame):
    """Handle interruption signals to save checkpoint before exit."""
    print(f"\n⚠️ Interruption received (signal {signum})")
    if checkpoint_file and processed_sessions:
        save_checkpoint(
            checkpoint_file, processed_sessions, current_session_idx, total_sessions
        )
    print("Exiting...")
    sys.exit(1)


def print_progress_bar(current, total, prefix="Progress", suffix="", length=50):
    """Print a progress bar to the console."""
    if total == 0:
        return

    percent = (current / total) * 100
    filled_length = int(length * current // total)
    bar = "█" * filled_length + "░" * (length - filled_length)

    sys.stdout.write(f"\r{prefix}: [{bar}] {percent:.1f}% {suffix}")
    sys.stdout.flush()

    if current == total:
        print()  # New line when complete


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
        analyze_detection_performance, run_adv_eval = get_evaluation_functions()
        if analyze_detection_performance is None:
            print("Warning: Evaluation functions not available")
            return

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
        analyze_detection_performance, _ = get_evaluation_functions()
        if analyze_detection_performance is None:
            print("Warning: Evaluation functions not available")
            return

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
        _, run_adv_eval = get_evaluation_functions()
        if run_adv_eval is None:
            print("Warning: Advanced evaluation functions not available")
            return

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


def cleanup_audio_files(data_path, session_name):
    """
    Supprime les fichiers audio découpés d'un site après traitement.

    Args:
        data_path: Chemin vers le répertoire des données audio
        session_name: Nom de la session pour les logs
    """
    try:
        audio_files = []
        # Trouver tous les fichiers audio dans le répertoire
        for ext in ["*.wav", "*.flac"]:
            audio_files.extend(glob.glob(os.path.join(data_path, ext)))

        if audio_files:
            print(
                f"  Suppression de {len(audio_files)} fichiers audio pour {session_name}"
            )

            # Supprimer les fichiers
            deleted_count = 0
            total_size = 0

            for audio_file in audio_files:
                try:
                    file_size = os.path.getsize(audio_file)
                    os.remove(audio_file)
                    deleted_count += 1
                    total_size += file_size
                except Exception as e:
                    print(f"    Erreur suppression {audio_file}: {e}")

            # Convertir la taille en unité lisible
            if total_size > 1024**3:
                size_str = f"{total_size / (1024**3):.2f} GB"
            elif total_size > 1024**2:
                size_str = f"{total_size / (1024**2):.2f} MB"
            else:
                size_str = f"{total_size / 1024:.2f} KB"

            print(f"    ✓ {deleted_count} fichiers supprimés ({size_str} libérés)")
        else:
            print(f"  Aucun fichier audio à supprimer pour {session_name}")

    except Exception as e:
        print(f"  Erreur lors du nettoyage pour {session_name}: {e}")


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
        return True, result_file  # Return success status
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
            "--multiprocessing",
            str(getattr(config, "multiprocessing", 1)),
            "--batch_size",
            str(getattr(config, "batch_size", 32)),
        ]

        try:
            # Run subprocess with real-time output to show progress bars
            result = subprocess.run(
                cmd,
                timeout=config.timeout,
                text=True,
                # Don't capture output to allow progress bars to show
                # but we'll check return code for success/failure
            )

            if result.returncode == 0:
                if os.path.exists(result_file):
                    print(f"✅ Success: {session_name}")
                    # Run site-level evaluation if we have the result file and annotations
                    if (
                        os.path.exists(result_file)
                        and annotation_dirs
                        and config.is_evaluation_mode
                    ):
                        run_site_level_evaluation(
                            result_file, session_name, annotation_dirs, config
                        )

                    # Clean up audio files if enabled in config and processing was successful
                    if os.path.exists(result_file) and getattr(
                        config, "cleanup_audio_files", False
                    ):
                        cleanup_audio_files(data_path, session_name)

                    return True, result_file if os.path.exists(result_file) else None

                    # return True, result_file  # Return success status
                else:
                    print(f"❌ Error: Result file not found for {session_name}")
                    return False, None
            else:
                print(
                    f"❌ Error processing {session_name}: Process failed with return code {result.returncode}"
                )
                return False, None

        except subprocess.TimeoutExpired:
            print(f"Timeout processing {session_name}")
            return False, None
        except Exception as e:
            print(f"❌ Exception processing {session_name}: {e}")
            return False, None


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
        analyze_detection_performance, _ = get_evaluation_functions()
        if analyze_detection_performance is None:
            print("Warning: Evaluation functions not available")
            return

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

    # Set up signal handler for graceful interruption
    def signal_handler(signum, frame):
        print("\n⚠️  Received interrupt signal. Saving checkpoint...")
        if "processed_sessions" in locals() and "config" in locals():
            save_checkpoint(
                config.output_base,
                processed_sessions,
                current_session_idx,
                total_sessions,
            )
        print("Checkpoint saved. Exiting...")
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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

        # Load checkpoint if exists
        checkpoint_path = os.path.join(config.output_base, "processing_checkpoint.json")
        if os.path.exists(checkpoint_path):
            checkpoint_data = load_checkpoint(config.output_base)
            processed_sessions = set(checkpoint_data.get("processed_sessions", []))
            current_session_idx = checkpoint_data.get("current_session_idx", 0)
            total_sessions_from_checkpoint = checkpoint_data.get("total_sessions", 0)
            print(
                f"Resuming from checkpoint: {len(processed_sessions)}/{total_sessions_from_checkpoint} sessions processed"
            )
        else:
            processed_sessions = set()
            current_session_idx = 0

        # Set global checkpoint path for signal handler
        checkpoint_file = config.output_base

        # Process each directory with site-level evaluation
        result_files = []
        total_sessions = len(data_dirs)

        for i, data_dir in enumerate(data_dirs):
            session_name = data_dir["name"]

            # Skip if already processed
            if session_name in processed_sessions:
                print(f"⏭️  Skipping already processed: {session_name}")
                result_file = os.path.join(
                    config.output_base, session_name, f"indices_{session_name}.csv"
                )
                if os.path.exists(result_file):
                    result_files.append(result_file)
                continue

            # Update progress
            current_session_idx = i
            print_progress_bar(
                i, total_sessions, prefix="Processing:", suffix=f"{session_name}"
            )

            # Process the directory
            success, result_file = run_detection_for_directory(
                data_dir, config, annotation_dirs
            )
            if success and result_file:
                result_files.append(result_file)
                processed_sessions.add(session_name)
                print(f"✓ Site processing completed: {session_name}")

                # Save checkpoint after successful processing
                save_checkpoint(
                    config.output_base, processed_sessions, i + 1, total_sessions
                )
            else:
                print(f"✗ Failed to process: {session_name}")

        # Final progress update
        print_progress_bar(
            total_sessions, total_sessions, prefix="Processing:", suffix="Complete"
        )

        # Clean up checkpoint file on successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("Checkpoint file cleaned up")

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
