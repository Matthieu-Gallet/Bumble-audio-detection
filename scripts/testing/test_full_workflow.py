#!/usr/bin/env python3
"""
Test script to verify the complete automated workflow functionality.
"""

import os
import sys
import pandas as pd
import glob
from pathlib import Path
import subprocess
import time


def test_folder_discovery():
    """Test folder discovery."""
    print("TEST 1: Folder discovery")
    print("-" * 40)

    try:
        # Import functions from main script
        import sys
        import os

        # Add scripts directory to path to import batch_process
        current_dir = os.path.dirname(os.path.abspath(__file__))
        scripts_dir = os.path.dirname(current_dir)
        sys.path.append(scripts_dir)

        from workflow.final_batch_process import (
            find_data_directories,
            find_annotation_directories,
        )

        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        BASE_DATA_PATH = os.path.join(parent_dir, "data")

        # Test data folder discovery
        data_dirs = find_data_directories(BASE_DATA_PATH)
        print(f"Data folders found: {len(data_dirs)}")

        if data_dirs:
            print("Data folders:")
            for i, d in enumerate(data_dirs[:5], 1):  # Limit to 5 for display
                print(
                    f"  {i}. {d['session']}/{d['subdirectory']} ({d['audio_count']} files)"
                )
            if len(data_dirs) > 5:
                print(f"  ... and {len(data_dirs) - 5} others")

        # Test annotation folder discovery
        annotation_dirs = find_annotation_directories(BASE_DATA_PATH)
        print(f"Annotation folders found: {len(annotation_dirs)}")

        if annotation_dirs:
            print("Annotation folders:")
            for i, d in enumerate(annotation_dirs[:5], 1):
                print(
                    f"  {i}. {d['session']}/{d['subdirectory']} ({d['annotation_count']} files)"
                )
            if len(annotation_dirs) > 5:
                print(f"  ... and {len(annotation_dirs) - 5} others")

        return len(data_dirs) > 0 and len(annotation_dirs) > 0

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_existing_results():
    """Test existing results verification."""
    print("\nTEST 2: Existing results verification")
    print("-" * 40)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    OUTPUT_BASE = os.path.join(parent_dir, "output_batch")

    if not os.path.exists(OUTPUT_BASE):
        print(f"Output folder does not exist yet: {OUTPUT_BASE}")
        return True

    # Check existing CSV files
    csv_files = glob.glob(os.path.join(OUTPUT_BASE, "*/indices_*.csv"))
    print(f"Existing CSV files: {len(csv_files)}")

    if csv_files:
        print("Files found:")
        for i, csv_file in enumerate(csv_files[:5], 1):
            session_name = os.path.basename(os.path.dirname(csv_file))
            print(f"  {i}. {session_name}")
        if len(csv_files) > 5:
            print(f"  ... and {len(csv_files) - 5} others")

    # Check merged file
    merged_file = os.path.join(OUTPUT_BASE, "merged_results.csv")
    if os.path.exists(merged_file):
        df = pd.read_csv(merged_file)
        print(f"Merged file found: {len(df)} rows")
    else:
        print("Merged file not found")

    # Check evaluation results
    eval_dirs = glob.glob(os.path.join(OUTPUT_BASE, "evaluation_results/*"))
    print(f"Evaluation directories: {len(eval_dirs)}")

    return True


def test_process_script():
    """Test the process.py script."""
    print("\nTEST 3: Process script test")
    print("-" * 40)

    try:
        # Take the first data directory
        # batch_process already imported above

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        from workflow.final_batch_process import find_data_directories

        BASE_DATA_PATH = os.path.join(parent_dir, "data")

        data_dirs = find_data_directories(BASE_DATA_PATH)

        if not data_dirs:
            print("No data directories found")
            return False

        # Take first directory
        test_dir = data_dirs[0]
        print(f"Test on: {test_dir['session']}/{test_dir['subdirectory']}")

        # Check if process.py exists
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        PROCESS_SCRIPT = os.path.join(parent_dir, "process.py")
        if not os.path.exists(PROCESS_SCRIPT):
            print(f"Process script not found: {PROCESS_SCRIPT}")
            return False

        print("Process script found")

        # Check virtual environment
        PYTHON_PATH = os.path.join(parent_dir, ".venv", "bin", "python")
        if not os.path.exists(PYTHON_PATH):
            print(f"Virtual environment not found: {PYTHON_PATH}")
            return False

        print("Virtual environment found")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_evaluation_functions():
    """Test evaluation functions."""
    print("\nTEST 4: Test evaluation functions")
    print("-" * 40)

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))

        # Check that evaluation scripts exist
        eval_scripts = [
            os.path.join(parent_dir, "scripts", "evaluation", "evaluate_detection.py"),
            os.path.join(parent_dir, "scripts", "evaluation", "advanced_evaluation.py"),
        ]

        for script in eval_scripts:
            if os.path.exists(script):
                print(f"✅ {os.path.basename(script)} found")
            else:
                print(f"❌ {os.path.basename(script)} missing")
                return False

        # Test import of advanced analysis functions
        # workflow.final_batch_process already imported above
        from workflow.final_batch_process import (
            find_optimal_threshold,
            analyze_false_positives_classes,
            run_analysis_with_optimal_threshold,
        )

        print("✅ Advanced analysis functions imported successfully")

        # Test with mock data
        import pandas as pd
        import numpy as np

        # Create test data
        np.random.seed(42)
        test_data = pd.DataFrame(
            {
                "score": np.random.random(100),
                "ground_truth": np.random.choice([0, 1], 100),
                "filename": [f"test_{i}.wav" for i in range(100)],
                "start_time": np.arange(100) * 5,
                "prediction": np.random.choice([0, 1], 100),
            }
        )

        # Test optimal threshold function
        optimal_result = find_optimal_threshold(test_data)
        print(
            f"✅ Optimal threshold calculated: {optimal_result['optimal_threshold']:.3f}"
        )

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_output_structure():
    """Test output directory structure."""
    print("\nTEST 5: Test output structure")
    print("-" * 40)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    OUTPUT_BASE = os.path.join(parent_dir, "output_batch")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    print(f"Output directory created/verified: {OUTPUT_BASE}")

    # Check write permissions
    test_file = os.path.join(OUTPUT_BASE, "test_write.txt")
    try:
        with open(test_file, "w") as f:
            f.write("Write test")
        os.remove(test_file)
        print("✅ Write permissions OK")
    except Exception as e:
        print(f"❌ Write error: {e}")
        return False

    return True


def run_mini_workflow():
    """Run a mini workflow to test the entire system."""
    print("\n🔍 TEST 6: Complete mini workflow")
    print("-" * 40)

    try:
        # Run the discovery script
        print("🚀 Launching discovery script...")

        # Use relative path for test_discovery.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_discovery_path = os.path.join(current_dir, "test_discovery.py")

        result = subprocess.run(
            [sys.executable, test_discovery_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("✅ Discovery script executed successfully")
            print("📊 Output:")
            print(
                result.stdout[:500] + "..."
                if len(result.stdout) > 500
                else result.stdout
            )
        else:
            print(f"❌ Error in discovery script: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ Discovery script timeout")
        return False
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return False

    return True


def main():
    """Main test function."""
    print("🧪 AUTOMATION SYSTEM TESTS")
    print("=" * 60)

    tests = [
        ("Folder discovery", test_folder_discovery),
        ("Existing results", test_existing_results),
        ("Process.py script", test_process_script),
        ("Evaluation functions", test_evaluation_functions),
        ("Output structure", test_output_structure),
        ("Mini workflow", run_mini_workflow),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))

    # Final summary
    print("\n📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<50} {status}")

    print(f"\n🎯 FINAL RESULT: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The system is ready.")
        print("\n📝 NEXT STEPS:")
        print("1. Run: python scripts/workflow/final_batch_process.py")
        print("2. Verify: python scripts/utilities/check_results.py")
        print("3. Results in: output_batch/")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
