#!/usr/bin/env python3
"""
Main launcher script for the complete audio detection automation system.
This script provides easy access to all system components.
"""

import os
import sys
import subprocess


def show_menu():
    """Display the main menu."""
    print("AUDIO DETECTION AUTOMATION SYSTEM")
    print("=" * 50)
    print("1. Run complete automation workflow")
    print("2. Run tests and system check")
    print("3. Check existing results")
    print("4. Clean previous results")
    print("5. Monitor workflow progress")
    print("6. View system documentation")
    print("7. Demo error analysis")
    print("8. Exit")
    print()


def run_workflow():
    """Run the complete automation workflow."""
    script_path = "scripts/workflow/run_full_automation.py"
    subprocess.run([sys.executable, script_path])


def run_tests():
    """Run system tests."""
    script_path = "scripts/testing/test_full_workflow.py"
    subprocess.run([sys.executable, script_path])


def check_results():
    """Check existing results."""
    script_path = "scripts/utilities/check_results.py"
    subprocess.run([sys.executable, script_path])


def clean_results():
    """Clean previous results."""
    script_path = "scripts/utilities/clean_results.py"
    subprocess.run([sys.executable, script_path])


def monitor_workflow():
    """Monitor workflow progress."""
    script_path = "scripts/utilities/monitor_workflow.py"
    subprocess.run([sys.executable, script_path])


def demo_error_analysis():
    """Run error analysis demonstration."""
    script_path = "scripts/testing/demo_error_analysis.py"
    subprocess.run([sys.executable, script_path])


def main():
    """Main function."""
    while True:
        try:
            show_menu()
            choice = input("Choose an option (1-8): ").strip()

            if choice == "1":
                run_workflow()
            elif choice == "2":
                run_tests()
            elif choice == "3":
                check_results()
            elif choice == "4":
                clean_results()
            elif choice == "5":
                monitor_workflow()
            elif choice == "6":
                # View documentation
                script_path = "scripts/utilities/view_docs.py"
                subprocess.run([sys.executable, script_path])
            elif choice == "7":
                demo_error_analysis()
            elif choice == "8":
                print("Goodbye!")
                break
            else:
                print("Invalid option. Please choose 1-8.")

            input("\nPress Enter to continue...")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
