#!/usr/bin/env python3
"""
Quick launch script for complete automation.
"""

import os
import sys
import subprocess
import time
from datetime import datetime


def print_header():
    """Display script header."""
    print("COMPLETE AUDIO DETECTION AUTOMATION")
    print("=" * 60)
    print(f"📅 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def check_prerequisites():
    """Check prerequisites."""
    print("🔍 Checking prerequisites...")

    # Check virtual environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    venv_path = os.path.join(parent_dir, ".venv", "bin", "python")
    if not os.path.exists(venv_path):
        print(f"Virtual environment missing: {venv_path}")
        return False
    print("Virtual environment found")

    # Check main scripts
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))

    scripts = [
        os.path.join(parent_dir, "process.py"),
        # os.path.join(os.path.dirname(current_dir), "batch_process.py"),
        os.path.join(current_dir, "..", "workflow", "final_batch_process.py"),
        os.path.join(current_dir, "..", "evaluation", "evaluate_detection.py"),
    ]

    for script in scripts:
        if not os.path.exists(script):
            print(f"Missing script: {script}")
            return False
    print("Main scripts found")

    # Check data directory
    data_path = os.path.join(parent_dir, "data")
    if not os.path.exists(data_path):
        print(f"Data directory missing: {data_path}")
        return False
    print("Data directory found")

    return True


def run_tests():
    """Run preliminary tests."""
    print("\nRunning preliminary tests...")

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        test_script = os.path.join(
            parent_dir, "scripts", "testing", "test_full_workflow.py"
        )

        result = subprocess.run(
            [
                sys.executable,
                test_script,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            print("Preliminary tests successful")
            return True
        else:
            print("Preliminary tests failed")
            print("Error:", result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("Preliminary tests timeout")
        return False
    except Exception as e:
        print(f"Error during tests: {e}")
        return False


def run_main_workflow():
    """Run main workflow."""
    print("\nLaunching main workflow...")

    try:
        # Launch main script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        batch_process_path = os.path.join(
            # os.path.dirname(current_dir), "batch_process.py"
            os.path.join(current_dir, "final_batch_process.py")
        )

        # Ask for configuration options
        config_response = (
            input("\nUse custom configuration file? (y/N): ").strip().lower()
        )
        cmd = [sys.executable, batch_process_path]

        if config_response in ["y", "yes", "oui"]:
            config_file = input(
                "Enter path to config file (or press Enter for default): "
            ).strip()
            if config_file:
                cmd.extend(["--config", config_file])

        # Ask for analysis mode
        mode_response = (
            input("\nAnalysis mode (individual/combined/both) [both]: ").strip().lower()
        )
        if mode_response in ["individual", "combined", "both"]:
            cmd.extend(["--analysis-mode", mode_response])

        result = subprocess.run(
            cmd,
            timeout=7200,
        )  # 2 hour timeout

        if result.returncode == 0:
            print("Main workflow completed successfully")
            return True
        else:
            print("Main workflow failed")
            return False

    except subprocess.TimeoutExpired:
        print("Main workflow timeout (2h)")
        return False
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user")
        return False
    except Exception as e:
        print(f"Error during workflow: {e}")
        return False


def show_results():
    """Show final results."""
    print("\nChecking results...")

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        check_script = os.path.join(
            parent_dir, "scripts", "utilities", "check_results.py"
        )

        result = subprocess.run(
            [
                sys.executable,
                check_script,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("Results generated successfully")
            print("\n" + "=" * 60)
            print(result.stdout)
            print("=" * 60)
        else:
            print("Error checking results")
            print(result.stderr)

    except Exception as e:
        print(f"Error displaying results: {e}")


def main():
    """Main function."""
    print_header()

    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\nPrerequisites not met. Stopping script.")
        return False

    # Step 2: Run tests (optional)
    run_tests_response = input("\nRun preliminary tests? (y/N): ").strip().lower()
    if run_tests_response in ["y", "yes", "oui"]:
        if not run_tests():
            continue_response = (
                input("\nTests failed. Continue anyway? (y/N): ").strip().lower()
            )
            if continue_response not in ["y", "yes", "oui"]:
                print("Stopping script.")
                return False

    # Step 3: Confirmation before launch
    print("\nTHE WORKFLOW WILL:")
    print("1. Process all audio directories with process.py")
    print("2. Merge all CSV results")
    print("3. Create combined ground truth")
    print("4. Evaluate performance with optimal thresholds")
    print("5. Generate detailed graphs and metrics")
    print("6. Analyze errors by class")

    confirm = input("\nLaunch complete workflow? (y/N): ").strip().lower()
    if confirm not in ["y", "yes", "oui"]:
        print("Workflow cancelled.")
        return False

    # Step 4: Launch workflow
    start_time = time.time()

    success = run_main_workflow()

    end_time = time.time()
    duration = end_time - start_time

    print(f"\nTotal duration: {duration/60:.1f} minutes")

    # Step 5: Show results
    if success:
        show_results()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        output_path = os.path.join(parent_dir, "output_batch")

        print("\nWORKFLOW COMPLETED SUCCESSFULLY!")
        print(f"� Results in: {output_path}")
        print("📊 Check evaluation directories for detailed analysis")

    else:
        print("\nWORKFLOW FAILED!")
        print("📝 Check error messages above")

    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
