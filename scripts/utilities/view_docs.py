#!/usr/bin/env python3
"""
Documentation viewer for the Audio Detection Automation System.
"""

import os
import sys
from pathlib import Path


def show_main_readme():
    """Show the main README file."""
    readme_path = "README.md"
    if os.path.exists(readme_path):
        print("📖 MAIN DOCUMENTATION")
        print("=" * 50)
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
            print(content)
    else:
        print("❌ Main README.md not found")


def show_config_docs():
    """Show configuration documentation."""
    config_readme = "config/README.md"
    if os.path.exists(config_readme):
        print("\n⚙️  CONFIGURATION DOCUMENTATION")
        print("=" * 50)
        with open(config_readme, "r", encoding="utf-8") as f:
            content = f.read()
            print(content)
    else:
        print("❌ Configuration README not found")


def show_workflow_docs():
    """Show workflow-specific documentation."""
    workflow_files = [
        "REFACTORING_FINAL_SUCCESS.md",
        "REFACTORING_FINAL_COMPLETE.md",
        "BATCH_PROCESS_MOVE_SUMMARY.md",
    ]

    for doc_file in workflow_files:
        if os.path.exists(doc_file):
            print(f"\n📋 {doc_file.upper()}")
            print("=" * 50)
            with open(doc_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Show only first 500 characters for summary
                if len(content) > 500:
                    content = content[:500] + "...\n[Full content available in file]"
                print(content)
            break


def show_usage_examples():
    """Show usage examples."""
    print("\n🚀 USAGE EXAMPLES")
    print("=" * 50)
    print(
        """
## Quick Start
1. Run setup: python setup.py
2. Start workflow: python main.py
3. Choose option 1 for full automation

## Configuration
1. Edit config/workflow_config.yaml
2. Set your data paths and preferences
3. Run: python scripts/workflow/final_batch_process.py --print-config

## Command Line Usage
# Run with custom config:
python scripts/workflow/final_batch_process.py --config my_config.yaml

# Run specific analysis:
python scripts/workflow/final_batch_process.py --analysis-mode individual --columns tag_Buzz

# Skip processing, only evaluate:
python scripts/workflow/final_batch_process.py --skip-processing

## Monitoring
# Quick status check:
python scripts/utilities/monitor_workflow.py

# Continuous monitoring:
python scripts/utilities/monitor_workflow.py --watch

# Check results:
python scripts/utilities/check_results.py
"""
    )


def show_menu():
    """Show documentation menu."""
    print("📚 AUDIO DETECTION SYSTEM DOCUMENTATION")
    print("=" * 50)
    print("1. Main README")
    print("2. Configuration Guide")
    print("3. Workflow Documentation")
    print("4. Usage Examples")
    print("5. Exit")
    print()


def main():
    """Main documentation viewer."""
    while True:
        try:
            show_menu()
            choice = input("Choose documentation section (1-5): ").strip()

            print("\n")

            if choice == "1":
                show_main_readme()
            elif choice == "2":
                show_config_docs()
            elif choice == "3":
                show_workflow_docs()
            elif choice == "4":
                show_usage_examples()
            elif choice == "5":
                print("📖 Documentation viewer closed.")
                break
            else:
                print("Invalid option. Please choose 1-5.")

            if choice in ["1", "2", "3", "4"]:
                input("\nPress Enter to continue...")

        except KeyboardInterrupt:
            print("\n\n📖 Documentation viewer closed.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
