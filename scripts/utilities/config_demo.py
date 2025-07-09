#!/usr/bin/env python3
"""
Configuration demo script for the Audio Detection Automation System.
This script demonstrates all configuration options and shows how to use them.
"""

import os
import sys
import argparse
from pathlib import Path

# Add config to path
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
)

try:
    from config_loader import WorkflowConfig

    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False
    print("❌ Configuration system not available")
    sys.exit(1)


def show_config_overview():
    """Show configuration overview."""
    print("⚙️  CONFIGURATION SYSTEM OVERVIEW")
    print("=" * 60)

    config = WorkflowConfig()

    print("\n📂 PATH CONFIGURATION:")
    print(f"  Data directory: {config.paths.data_dir}")
    print(f"  Python executable: {config.paths.python_path}")
    print(f"  Process script: {config.paths.process_script}")
    print(f"  Output directory: {config.paths.output_dir}")

    print("\n🔧 PROCESSING CONFIGURATION:")
    print(f"  Segment length: {config.processing.segment_length} seconds")
    print(f"  Audio format: {config.processing.audio_format}")
    print(f"  Timeout: {config.processing.timeout} seconds")

    print("\n📊 ANALYSIS CONFIGURATION:")
    print(f"  Mode: {config.analysis.mode}")
    print(f"  Columns: {', '.join(config.analysis.columns)}")
    print(f"  Default threshold: {config.analysis.default_threshold}")
    print(f"  Evaluation duration: {config.analysis.evaluation_duration}")

    print("\n📋 SESSION CONFIGURATION:")
    if config.sessions.target_sessions:
        print(f"  Target sessions: {', '.join(config.sessions.target_sessions)}")
    else:
        print("  Target sessions: All sessions")
    print(f"  Skip processing: {config.sessions.skip_processing}")

    print("\n📈 EVALUATION FEATURES:")
    features = config.evaluation.features
    print(f"  Optimal threshold: {features.optimal_threshold}")
    print(f"  Error analysis: {features.error_analysis}")
    print(f"  Class breakdown: {features.class_breakdown}")
    print(f"  Visualizations: {features.visualizations}")
    print(f"  Cross-session comparison: {features.cross_session_comparison}")

    print("\n🔍 THRESHOLD SEARCH:")
    search = config.evaluation.threshold_search
    print(f"  Range: {search.min_threshold} to {search.max_threshold}")
    print(f"  Number of thresholds: {search.num_thresholds}")

    print("\n📊 VISUALIZATION SETTINGS:")
    viz = config.evaluation.visualization
    print(f"  DPI: {viz.dpi}")
    print(f"  Figure size: {viz.figure_size}")
    print(f"  Save format: {viz.save_format}")

    print("\n💾 OUTPUT CONFIGURATION:")
    formats = config.output.formats
    print(f"  CSV: {formats.csv}")
    print(f"  JSON: {formats.json}")
    print(f"  Plots: {formats.plots}")
    print(f"  Decimal places: {config.output.decimal_places}")

    print("\n🔧 ADVANCED SETTINGS:")
    if config.advanced.parallel.enabled:
        print(f"  Parallel processing: {config.advanced.parallel.max_workers} workers")
    else:
        print("  Parallel processing: Disabled")
    print(f"  Continue on error: {config.advanced.error_handling.continue_on_error}")
    print(f"  Max retries: {config.advanced.error_handling.max_retries}")


def show_usage_examples():
    """Show usage examples with different configurations."""
    print("\n🚀 USAGE EXAMPLES")
    print("=" * 60)

    print("\n1. DEFAULT CONFIGURATION:")
    print("   python scripts/workflow/final_batch_process.py")
    print("   → Uses config/workflow_config.yaml settings")

    print("\n2. CUSTOM CONFIG FILE:")
    print("   python scripts/workflow/final_batch_process.py --config my_config.yaml")
    print("   → Uses custom configuration file")

    print("\n3. OVERRIDE ANALYSIS MODE:")
    print(
        "   python scripts/workflow/final_batch_process.py --analysis-mode individual"
    )
    print("   → Run only individual session analysis")

    print("\n4. SPECIFIC COLUMNS:")
    print(
        "   python scripts/workflow/final_batch_process.py --columns tag_Buzz tag_Insect"
    )
    print("   → Analyze only specific detection columns")

    print("\n5. SPECIFIC SESSIONS:")
    print(
        "   python scripts/workflow/final_batch_process.py --sessions 20240408_session_01_Tent_SM05_T"
    )
    print("   → Process only specific sessions")

    print("\n6. SKIP PROCESSING:")
    print("   python scripts/workflow/final_batch_process.py --skip-processing")
    print("   → Skip audio processing, only run evaluation")

    print("\n7. COMBINED OPTIONS:")
    print("   python scripts/workflow/final_batch_process.py \\")
    print("     --analysis-mode both \\")
    print("     --columns tag_Buzz biophony \\")
    print(
        "     --sessions 20240408_session_01_Tent_SM05_T 20240612_session_02_Tent_SM06_T"
    )
    print("   → Multiple overrides combined")


def show_config_creation():
    """Show how to create custom configurations."""
    print("\n🛠️  CREATING CUSTOM CONFIGURATIONS")
    print("=" * 60)

    print("\n1. COPY EXAMPLE CONFIG:")
    print("   cp config/example_config.yaml config/my_config.yaml")

    print("\n2. EDIT YOUR CONFIG:")
    print("   # Common customizations:")
    print("   paths:")
    print('     data_dir: "/path/to/my/data"')
    print('     output_dir: "/path/to/my/output"')
    print("   ")
    print("   analysis:")
    print('     mode: "individual"')
    print('     columns: ["tag_Buzz", "tag_Bird"]')
    print("   ")
    print("   sessions:")
    print('     target_sessions: ["session1", "session2"]')
    print("     skip_processing: false")

    print("\n3. USE YOUR CONFIG:")
    print("   python scripts/workflow/final_batch_process.py --config my_config.yaml")

    print("\n4. VALIDATE CONFIG:")
    print(
        "   python scripts/workflow/final_batch_process.py --config my_config.yaml --print-config"
    )


def show_troubleshooting():
    """Show troubleshooting tips."""
    print("\n🔧 TROUBLESHOOTING")
    print("=" * 60)

    print("\n❌ CONFIG FILE NOT FOUND:")
    print("   • Check if config/workflow_config.yaml exists")
    print("   • Copy from config/example_config.yaml if needed")
    print("   • Use absolute paths in config file")

    print("\n❌ IMPORT ERROR:")
    print("   • Ensure PyYAML is installed: pip install pyyaml")
    print("   • Check Python path includes config directory")

    print("\n❌ PATH ERRORS:")
    print("   • Use absolute paths in configuration")
    print("   • Check that data directory exists")
    print("   • Verify Python executable path")

    print("\n❌ PROCESSING ERRORS:")
    print("   • Check session folder structure")
    print("   • Verify audio files exist (.wav format)")
    print("   • Monitor workflow: python scripts/utilities/monitor_workflow.py")

    print("\n💡 DEBUGGING TIPS:")
    print("   • Use --print-config to verify settings")
    print("   • Start with individual mode for testing")
    print("   • Check logs in output directory")
    print("   • Use skip-processing to test evaluation only")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Configuration system demonstration")
    parser.add_argument(
        "--section",
        choices=["overview", "examples", "creation", "troubleshooting"],
        help="Show specific section",
    )

    args = parser.parse_args()

    if not USE_CONFIG:
        return

    print("🎵 AUDIO DETECTION CONFIGURATION DEMO")
    print("=" * 60)

    if args.section == "overview":
        show_config_overview()
    elif args.section == "examples":
        show_usage_examples()
    elif args.section == "creation":
        show_config_creation()
    elif args.section == "troubleshooting":
        show_troubleshooting()
    else:
        # Show all sections
        show_config_overview()
        show_usage_examples()
        show_config_creation()
        show_troubleshooting()

    print("\n📚 For more information, see config/README.md")


if __name__ == "__main__":
    main()
