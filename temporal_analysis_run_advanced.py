#!/usr/bin/env python3
"""
Script pour exécuter l'évaluation avancée globale sur tous les dossiers output_batch_XX
"""
import os
import subprocess
import yaml


def main():
    """Exécute l'évaluation avancée pour chaque dossier output_batch_XX."""

    base_dir = "~/Documents/Acoustique/detection"
    os.chdir(base_dir)

    # Find all output_batch_XX directories
    output_dirs = []
    for item in os.listdir(base_dir):
        if item.startswith("output_batch_") and os.path.isdir(item):
            output_dirs.append(item)

    output_dirs.sort(key=lambda x: int(x.split("_")[-1]))  # Sort by duration

    print(f"Found {len(output_dirs)} output directories: {output_dirs}")

    for output_dir in output_dirs:
        print(f"\n{'='*60}")
        print(f"Processing {output_dir}")
        print(f"{'='*60}")

        # Vérifier que le fichier merged_results.csv existe
        merged_file = os.path.join(output_dir, "merged_results.csv")
        if not os.path.exists(merged_file):
            print(f"❌ No merged_results.csv found in {output_dir}")
            continue

        # Créer un config temporaire pour ce dossier
        config_temp = f"temp_config_{output_dir}.yaml"

        try:
            # Lire le config de base
            with open("scripts/config/simple_config.yaml", "r") as f:
                config_data = yaml.safe_load(f)

            # Modifier les paramètres nécessaires
            config_data["paths"]["output_dir"] = output_dir
            config_data["analysis"]["mode"] = "evaluation"

            # Extraire la durée du nom du dossier
            duration = int(output_dir.split("_")[-1])
            config_data["processing"]["segment_length"] = duration
            config_data["advanced_evaluation"]["duration"] = duration
            config_data["advanced_evaluation"][
                "output_dir"
            ] = f"{output_dir}/global_advanced_evaluation"

            # Écrire le config temporaire
            with open(config_temp, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False)

            # Exécuter process_batch.py avec skip-processing et evaluation mode
            cmd = [
                "python",
                "process_batch.py",
                "--config",
                config_temp,
                "--mode",
                "evaluation",
                "--skip-processing",
            ]

            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, text=True)

            if result.returncode == 0:
                print(f"✅ Advanced evaluation completed for {output_dir}")
            else:
                print(f"❌ Error in {output_dir} (return code: {result.returncode})")

        except Exception as e:
            print(f"❌ Exception processing {output_dir}: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Nettoyer le config temporaire
            if os.path.exists(config_temp):
                os.remove(config_temp)

    print(f"\n{'='*60}")
    print("Advanced evaluation completed for all directories!")


if __name__ == "__main__":
    main()
