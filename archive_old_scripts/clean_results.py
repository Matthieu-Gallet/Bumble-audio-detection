#!/usr/bin/env python3
"""
Script pour nettoyer les résultats précédents et redémarrer à zéro.
"""

import os
import shutil
import glob
from pathlib import Path


def clean_output_directory():
    """Clean main output directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    OUTPUT_BASE = os.path.join(parent_dir, "output_batch")

    print("🧹 Cleaning output directory...")

    if os.path.exists(OUTPUT_BASE):
        try:
            # Lister le contenu avant suppression
            contents = os.listdir(OUTPUT_BASE)
            if contents:
                print(f"📁 Contenu à supprimer:")
                for item in contents:
                    item_path = os.path.join(OUTPUT_BASE, item)
                    if os.path.isdir(item_path):
                        file_count = len(
                            glob.glob(os.path.join(item_path, "**/*"), recursive=True)
                        )
                        print(f"  📁 {item}/ ({file_count} fichiers)")
                    else:
                        print(f"  📄 {item}")

                confirm = (
                    input("\n❓ Confirmer la suppression ? (y/N): ").strip().lower()
                )
                if confirm not in ["y", "yes", "oui"]:
                    print("❌ Nettoyage annulé")
                    return False

                # Supprimer le contenu
                shutil.rmtree(OUTPUT_BASE)
                print("✅ Dossier de sortie nettoyé")
            else:
                print("✅ Dossier de sortie déjà vide")

        except Exception as e:
            print(f"❌ Erreur lors du nettoyage: {e}")
            return False
    else:
        print("✅ Dossier de sortie n'existe pas")

    # Recréer le dossier
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    print(f"✅ Dossier de sortie recréé: {OUTPUT_BASE}")

    return True


def clean_temporary_files():
    """Nettoyer les fichiers temporaires."""
    print("\n🧹 Nettoyage des fichiers temporaires...")

    # Get project root dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(current_dir))

    temp_patterns = [
        "/tmp/audio_*.wav",
        "/tmp/detection_*.csv",
        os.path.join(project_dir, "temp_*"),
        os.path.join(project_dir, "*.tmp"),
    ]

    cleaned_count = 0

    for pattern in temp_patterns:
        temp_files = glob.glob(pattern)
        for temp_file in temp_files:
            try:
                if os.path.isfile(temp_file):
                    os.remove(temp_file)
                    cleaned_count += 1
                elif os.path.isdir(temp_file):
                    shutil.rmtree(temp_file)
                    cleaned_count += 1
            except Exception as e:
                print(f"⚠️  Unable to delete {temp_file}: {e}")

    print(f"✅ {cleaned_count} temporary files cleaned")
    return True


def clean_cache_files():
    """Clean Python cache files."""
    print("\n🧹 Cleaning Python cache files...")

    # Get project root dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(current_dir))
    cache_patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/.pytest_cache",
    ]

    cleaned_count = 0

    for pattern in cache_patterns:
        cache_files = glob.glob(os.path.join(project_dir, pattern), recursive=True)
        for cache_file in cache_files:
            try:
                if os.path.isfile(cache_file):
                    os.remove(cache_file)
                    cleaned_count += 1
                elif os.path.isdir(cache_file):
                    shutil.rmtree(cache_file)
                    cleaned_count += 1
            except Exception as e:
                print(f"⚠️  Unable to delete {cache_file}: {e}")

    print(f"✅ {cleaned_count} cache files cleaned")
    return True


def show_current_state():
    """Show current system state."""
    print("\n📊 Current system state:")
    print("-" * 40)

    # Get project root dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(current_dir))

    # Output directory
    OUTPUT_BASE = os.path.join(project_dir, "output_batch")
    if os.path.exists(OUTPUT_BASE):
        contents = os.listdir(OUTPUT_BASE)
        print(f"📁 Output directory: {len(contents)} elements")

        # Count CSV files
        csv_files = glob.glob(os.path.join(OUTPUT_BASE, "*/indices_*.csv"))
        print(f"📄 Result CSV files: {len(csv_files)}")

        # Check merged file
        merged_file = os.path.join(OUTPUT_BASE, "merged_results.csv")
        if os.path.exists(merged_file):
            print("📊 Merged file: ✅ Present")
        else:
            print("📊 Merged file: ❌ Missing")

        # Count evaluations
        eval_dirs = glob.glob(os.path.join(OUTPUT_BASE, "evaluation_*"))
        print(f"📈 Evaluation directories: {len(eval_dirs)}")

    else:
        print("📁 Output directory: ❌ Missing")

    # Data directory
    DATA_PATH = os.path.join(project_dir, "data")
    if os.path.exists(DATA_PATH):
        data_items = os.listdir(DATA_PATH)
        print(f"📁 Data directory: {len(data_items)} sessions")
    else:
        print("📁 Data directory: ❌ Missing")


def main():
    """Main function."""
    print("🧹 AUTOMATION SYSTEM CLEANUP")
    print("=" * 60)

    # Show current state
    show_current_state()

    # Cleanup options
    print("\n🛠️  Available cleanup options:")
    print("1. Clean output results")
    print("2. Clean temporary files")
    print("3. Clean Python cache files")
    print("4. Complete cleanup (all)")
    print("5. Cancel")

    choice = input("\n❓ Choose an option (1-5): ").strip()

    if choice == "1":
        clean_output_directory()
    elif choice == "2":
        clean_temporary_files()
    elif choice == "3":
        clean_cache_files()
    elif choice == "4":
        print("\n🧹 Complete cleanup in progress...")
        clean_output_directory()
        clean_temporary_files()
        clean_cache_files()
        print("\n✅ Complete cleanup finished")
    elif choice == "5":
        print("❌ Cleanup cancelled")
        return False
    else:
        print("❌ Invalid option")
        return False

    # Show state after cleanup
    print("\n📊 State after cleanup:")
    show_current_state()

    print("\n🎉 Cleanup finished! The system is ready for new processing.")

    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Nettoyage interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
