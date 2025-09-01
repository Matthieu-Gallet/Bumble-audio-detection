#!/usr/bin/env python3
"""
Setup script for the Audio Detection Automation System.
This script helps new users configure their environment and verify the setup.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_header():
    """Print the setup header."""
    print("AUDIO DETECTION AUTOMATION SYSTEM SETUP")
    print("=" * 50)
    print()


def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(
            f"❌ Python {version.major}.{version.minor}.{version.micro} (Minimum required: 3.8)"
        )
        return False


def check_requirements():
    """Check if required packages are installed."""
    print("\n📦 Checking required packages...")
    required_packages = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
        "librosa",
        "soundfile",
        "pyyaml",
        "plotly",
        "dash",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True


def check_directory_structure():
    """Check if the required directory structure exists."""
    print("\n📁 Checking directory structure...")
    required_dirs = ["data", "scripts", "utils", "output_batch"]
    required_files = ["process.py", "main.py", "requirements.txt"]

    all_good = True

    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/")
            all_good = False

    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name}")
            all_good = False

    return all_good


def setup_config():
    """Setup configuration files."""
    print("\n⚙️  Setting up configuration...")

    config_dir = Path("scripts/config")
    if not config_dir.exists():
        config_dir.mkdir()
        print("✅ Created scripts/config/ directory")

    # Check if config files exist
    config_file = config_dir / "workflow_config.yaml"
    example_config = config_dir / "example_config.yaml"

    if config_file.exists():
        print("✅ Configuration file exists")
    else:
        if example_config.exists():
            shutil.copy(example_config, config_file)
            print("✅ Created configuration from example")
        else:
            print("❌ No configuration files found")
            return False

    return True


def setup_virtual_environment():
    """Check and setup virtual environment."""
    print("\n🌐 Checking virtual environment...")

    venv_path = Path(".venv")
    if venv_path.exists():
        print("✅ Virtual environment exists")

        # Check if it's activated
        if hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        ):
            print("✅ Virtual environment is activated")
        else:
            print("⚠️  Virtual environment exists but not activated")
            print(
                "   Run: source .venv/bin/activate (Linux/Mac) or .venv\\Scripts\\activate (Windows)"
            )

        return True
    else:
        print("❌ Virtual environment not found")
        response = input("Create virtual environment? (y/n): ")
        if response.lower() == "y":
            try:
                subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
                print("✅ Virtual environment created")
                print("   Don't forget to activate it and install requirements!")
                return True
            except subprocess.CalledProcessError:
                print("❌ Failed to create virtual environment")
                return False
        return False


def check_data_structure():
    """Check if data directory has the expected structure."""
    print("\n📊 Checking data structure...")

    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found")
        print("   Create 'data/' directory and add your audio session folders")
        return False

    # Look for session folders
    session_folders = [
        d for d in data_dir.iterdir() if d.is_dir() and "session" in d.name.lower()
    ]

    if session_folders:
        print(f"✅ Found {len(session_folders)} session folders:")
        for folder in sorted(session_folders):
            print(f"   - {folder.name}")
    else:
        print("⚠️  No session folders found in data/")
        print("   Expected folders like: 20240408_session_01_Tent_SM05_T")

    return len(session_folders) > 0


def run_quick_test():
    """Run a quick test to verify everything works."""
    print("\n🧪 Running quick test...")

    try:
        # Try to import the config loader
        from scripts.config.config_loader import WorkflowConfig

        config = WorkflowConfig()
        print("✅ Configuration system works")

        # Try to run the help command
        result = subprocess.run(
            [sys.executable, "scripts/workflow/final_batch_process.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print("✅ Main workflow script accessible")
        else:
            print("❌ Main workflow script has issues")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    return True


def main():
    """Main setup function."""
    print_header()

    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_requirements),
        ("Directory Structure", check_directory_structure),
        ("Configuration Setup", setup_config),
        ("Virtual Environment", setup_virtual_environment),
        ("Data Structure", check_data_structure),
        ("Quick Test", run_quick_test),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 50)
    print(" SETUP SUMMARY")
    print("=" * 50)

    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
        if result:
            passed += 1

    print(f"\nResult: {passed}/{len(results)} checks passed")

    if passed == len(results):
        print("\n Setup complete! You can now run the system with:")
        print("   python main.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("   See README.md for detailed setup instructions.")


if __name__ == "__main__":
    main()
