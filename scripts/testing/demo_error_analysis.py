#!/usr/bin/env python3
"""
Demonstration script for the enhanced error analysis capabilities.
This script shows how to use the new detailed error analysis functions.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)


def create_demo_data():
    """Create demonstration data for error analysis."""
    print("Creating demonstration data...")

    # Create fake segments data
    np.random.seed(42)
    n_segments = 1000

    segments_data = {
        "filename": [f"demo_file_{i//10}.wav" for i in range(n_segments)],
        "start_time": [(i % 10) * 5 for i in range(n_segments)],
        "ground_truth": np.random.choice([0, 1], n_segments, p=[0.8, 0.2]),
        "score": np.random.random(n_segments),
        "prediction": np.random.choice([0, 1], n_segments, p=[0.7, 0.3]),
    }

    segments_df = pd.DataFrame(segments_data)

    # Create fake CSV data with multiple detection columns
    csv_data = {
        "name": segments_data["filename"],
        "start": segments_data["start_time"],
        "tag_Buzz": segments_data["score"],
        "tag_Bird": np.random.random(n_segments),
        "tag_Insect": np.random.random(n_segments),
        "tag_Mammal": np.random.random(n_segments),
        "tag_Wind": np.random.random(n_segments),
        "tag_Human": np.random.random(n_segments),
        "tag_Mechanical": np.random.random(n_segments),
    }

    csv_df = pd.DataFrame(csv_data)

    # Save demo data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    output_dir = os.path.join(parent_dir, "demo_analysis")
    os.makedirs(output_dir, exist_ok=True)

    segments_df.to_csv(os.path.join(output_dir, "demo_segments.csv"), index=False)
    csv_df.to_csv(os.path.join(output_dir, "demo_merged.csv"), index=False)

    print(f"Demo data created in: {output_dir}")
    print(f"Segments: {len(segments_df)}")
    print(
        f"Ground truth distribution: {segments_df['ground_truth'].value_counts().to_dict()}"
    )
    print(
        f"Prediction distribution: {segments_df['prediction'].value_counts().to_dict()}"
    )

    return segments_df, csv_df, output_dir


def run_error_analysis_demo():
    """Run demonstration of error analysis."""
    print("\nDEMONSTRATION: Enhanced Error Analysis")
    print("=" * 60)

    # Create demo data
    segments_df, csv_df, output_dir = create_demo_data()

    # Import the new error analysis function
    import sys

    # Add parent directory to path to import batch_process
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.dirname(current_dir)
    sys.path.append(scripts_dir)

    from batch_process import create_detailed_error_analysis_table

    # Run detailed error analysis
    print("\nRunning detailed error analysis...")

    merged_csv_path = os.path.join(output_dir, "demo_merged.csv")

    # Test the new error analysis function
    error_results = create_detailed_error_analysis_table(
        merged_csv_path, segments_df, "tag_Buzz", output_dir
    )

    print(f"\nError Analysis Results:")
    print(f"False Positives: {error_results['false_positives_count']}")
    print(f"False Negatives: {error_results['false_negatives_count']}")

    # List generated files
    print(f"\nGenerated Analysis Files:")
    analysis_files = [
        f for f in os.listdir(output_dir) if f.endswith(".csv") or f.endswith(".png")
    ]
    for file in sorted(analysis_files):
        print(f"  - {file}")

    # Show sample of detailed analysis
    if error_results["false_positives_count"] > 0:
        print(f"\nSample False Positive Analysis:")
        fp_sample = error_results["false_positives_data"][:3]
        for i, fp in enumerate(fp_sample, 1):
            print(
                f"  {i}. File: {fp['filename']}, Time: {fp['start_time']}, "
                f"Score: {fp['detection_score']:.3f}, Class: {fp['dominant_class']}"
            )

    if error_results["false_negatives_count"] > 0:
        print(f"\nSample False Negative Analysis:")
        fn_sample = error_results["false_negatives_data"][:3]
        for i, fn in enumerate(fn_sample, 1):
            print(
                f"  {i}. File: {fn['filename']}, Time: {fn['start_time']}, "
                f"Score: {fn['detection_score']:.3f}, Class: {fn['dominant_class']}"
            )

    print(f"\nDemo completed! Check {output_dir} for detailed results.")


def show_table_formats():
    """Show the format of generated error analysis tables."""
    print("\nTABLE FORMATS DEMONSTRATION")
    print("=" * 40)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    demo_dir = os.path.join(parent_dir, "demo_analysis")

    # Show False Positives detailed table format
    fp_detailed_file = os.path.join(demo_dir, "false_positives_detailed_tag_Buzz.csv")
    if os.path.exists(fp_detailed_file):
        print("\nFALSE POSITIVES DETAILED TABLE (first 5 rows):")
        fp_df = pd.read_csv(fp_detailed_file)
        print(fp_df.head().to_string(index=False))
        print(f"Columns: {list(fp_df.columns)}")

    # Show False Positives summary table format
    fp_summary_file = os.path.join(demo_dir, "false_positives_summary_tag_Buzz.csv")
    if os.path.exists(fp_summary_file):
        print(f"\nFALSE POSITIVES SUMMARY TABLE:")
        fp_summary = pd.read_csv(fp_summary_file)
        print(fp_summary.to_string())

    # Show combined error summary
    combined_file = os.path.join(demo_dir, "combined_error_summary_tag_Buzz.csv")
    if os.path.exists(combined_file):
        print(f"\nCOMBINED ERROR SUMMARY TABLE:")
        combined_df = pd.read_csv(combined_file)
        print(combined_df.to_string(index=False))


def main():
    """Main demonstration function."""
    print("ENHANCED ERROR ANALYSIS DEMONSTRATION")
    print("=" * 50)
    print("This script demonstrates the new error analysis capabilities")
    print("including detailed tables for false positives and false negatives.")

    # Run the demo
    run_error_analysis_demo()

    # Show table formats
    show_table_formats()

    print("\nKEY IMPROVEMENTS:")
    print("- Detailed error breakdown by dominant class")
    print("- Top 3 contributing classes for each error")
    print("- Statistical analysis (mean, std, min, max scores)")
    print("- Combined error summary tables")
    print("- Error visualization plots")
    print("- File-level error tracking")

    print("\nALL EMOJI CHARACTERS REMOVED FROM OUTPUT")
    print("ALL DOCUMENTATION TRANSLATED TO ENGLISH")
    print("FILES ORGANIZED IN STRUCTURED DIRECTORIES")


if __name__ == "__main__":
    main()
