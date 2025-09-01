#!/usr/bin/env python3
"""
Results checker for the Audio Detection Automation System.
This script provides a comprehensive overview of processing results.
"""

import os
import sys
import pandas as pd
import glob
import time
from pathlib import Path

# Add config to path
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
    )
)

try:
    from config_loader import WorkflowConfig

    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False
    print("⚠️  Configuration system not available, using default paths")


def get_output_path():
    """Get output path from config or default."""
    if USE_CONFIG:
        try:
            config = WorkflowConfig()
            return config.paths.output_dir
        except:
            pass

    # Fallback to default path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    return os.path.join(parent_dir, "output_batch")


def check_batch_results():
    """Check the results of automated processing."""

    OUTPUT_BASE = get_output_path()

    if not os.path.exists(OUTPUT_BASE):
        print(f"❌ Output directory not found: {OUTPUT_BASE}")
        print("   Run the workflow first to generate results.")
        return

    print("📊 AUTOMATION PROCESSING SUMMARY")
    print("=" * 50)

    # Check merged file
    merged_file = os.path.join(OUTPUT_BASE, "merged_results.csv")
    if os.path.exists(merged_file):
        try:
            df = pd.read_csv(merged_file)
            print(f"✅ Merged file: {len(df)} rows")

            # Show segments per session
            if "name" in df.columns:
                session_counts = df["name"].value_counts()
                print(f"\n📊 Segments per session:")
                for session, count in session_counts.items():
                    print(f"  - {session}: {count} segments")

            # Show available columns
            detection_columns = [
                col
                for col in df.columns
                if "tag_" in col or col in ["buzz", "biophony"]
            ]
            if detection_columns:
                print(f"\n🏷️  Detection columns: {', '.join(detection_columns)}")

        except Exception as e:
            print(f"❌ Error reading merged file: {e}")
    else:
        print("❌ Merged file not found")

    # Check evaluations
    eval_dirs = glob.glob(os.path.join(OUTPUT_BASE, "eval_*"))
    if eval_dirs:
        print(f"\n📈 Evaluations found: {len(eval_dirs)}")
        for eval_dir in sorted(eval_dirs):
            col_name = os.path.basename(eval_dir).replace("eval_", "")

            # Check metrics
            metrics_file = os.path.join(eval_dir, "metrics.txt")
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, "r") as f:
                        content = f.read()
                        if "f1_score:" in content:
                            f1_lines = [
                                l for l in content.split("\n") if "f1_score:" in l
                            ]
                            if f1_lines:
                                f1_value = f1_lines[0].split(":")[1].strip()
                                print(f"  - {col_name}: F1-Score = {f1_value}")
                except Exception as e:
                    print(f"  - {col_name}: Error reading metrics ({e})")
            else:
                print(f"  - {col_name}: No metrics file")
    else:
        print("\n❌ No evaluations found")

    # Check evaluation summary
    summary_file = os.path.join(OUTPUT_BASE, "evaluation_summary.csv")
    if os.path.exists(summary_file):
        try:
            print(f"\n🏆 PERFORMANCE SUMMARY:")
            summary_df = pd.read_csv(summary_file)
            print(summary_df.to_string(index=False, float_format="%.3f"))

            if len(summary_df) > 0:
                best_row = summary_df.loc[summary_df["f1_score"].idxmax()]
                print(
                    f"\n🥇 Best column: {best_row['column']} (F1: {best_row['f1_score']:.3f})"
                )
        except Exception as e:
            print(f"❌ Error reading summary: {e}")
    else:
        print("\n❌ No evaluation summary found")

    # Check for error analysis
    error_files = glob.glob(os.path.join(OUTPUT_BASE, "*error*"))
    if error_files:
        print(f"\n🔍 Error analysis files: {len(error_files)}")
        for error_file in sorted(error_files)[:5]:  # Show first 5
            print(f"  - {os.path.basename(error_file)}")
        if len(error_files) > 5:
            print(f"  ... and {len(error_files) - 5} more")

    # Show recent activity
    try:
        all_files = []
        for root, dirs, files in os.walk(OUTPUT_BASE):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    all_files.append((file_path, stat.st_mtime))

        if all_files:
            # Sort by modification time
            all_files.sort(key=lambda x: x[1], reverse=True)
            latest_file = all_files[0]
            latest_time = time.ctime(latest_file[1])
            print(f"\n🕒 Latest activity: {latest_time}")
            print(f"   File: {os.path.relpath(latest_file[0], OUTPUT_BASE)}")

    except Exception as e:
        print(f"\n⚠️  Could not check recent activity: {e}")

    print(f"\n📁 All files are in: {OUTPUT_BASE}")
    print("\n💡 Tip: Use 'python main.py' to run the full workflow")


if __name__ == "__main__":
    check_batch_results()
