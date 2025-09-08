#!/usr/bin/env python3
"""
Script de sélection et copie de données acoustiques
Critères de sélection :
- Enregistreurs : Loriaz 1600, Loriaz 2100, Peclerey 1400, Diosaz
- Période : 1er avril → 20 juin
- Heures : 8h00 → 21h00
"""

import os
import shutil
import re
from datetime import datetime, time
from pathlib import Path
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import time as time_module

# Configuration des sites cibles
SITES_CONFIG = {
    # Loriaz
    "SMA02964-D05": {
        "location": "LORIAZ-1630",
        "target": "Loriaz 1600 (voix)",
        "altitude": 1630,
    },
    "SMA02975-D08": {
        "location": "LORIAZ-2140",
        "target": "Loriaz 2100 (vent)",
        "altitude": 2140,
    },
    # Peclerey
    "SMA02961-D04": {
        "location": "PECLEREY-1400",
        "target": "Peclerey 1400 (Helico)",
        "altitude": 1400,
    },
    # Diosaz
    "SMA02939-D01": {
        "location": "SERVOZ-DIOSAZ",
        "target": "Diosaz (riviere)",
        "altitude": 838,
    },
}

# Dates limites
DATE_START = datetime(2025, 4, 1)  # 1er avril
DATE_END = datetime(2025, 6, 20)  # 20 juin

# Heures limites
TIME_START = time(8, 0)  # 8h00
TIME_END = time(21, 0)  # 21h00


def parse_filename(filename):
    """
    Parse un nom de fichier pour extraire la date et l'heure
    Format attendu : {SERIAL}_{YYYYMMDD}_{HHMMSS}.wav
    """
    try:
        # Extraire la partie date/heure du nom de fichier
        match = re.search(r"_(\d{8})_(\d{6})\.wav$", filename)
        if not match:
            return None, None

        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMMSS

        # Convertir en objets datetime
        file_date = datetime.strptime(date_str, "%Y%m%d")
        file_time = datetime.strptime(time_str, "%H%M%S").time()

        return file_date, file_time

    except Exception as e:
        print(f"Erreur parsing {filename}: {e}")
        return None, None


def check_file_criteria(filename, serial_number):
    """
    Vérifie si un fichier respecte les critères de sélection
    """
    # Vérifier si le serial number est dans nos sites cibles
    if serial_number not in SITES_CONFIG:
        return False, "Site non sélectionné"

    # Parser la date et l'heure
    file_date, file_time = parse_filename(filename)
    if file_date is None or file_time is None:
        return False, "Format de fichier invalide"

    # Vérifier la période
    if not (DATE_START <= file_date <= DATE_END):
        return False, f"Date hors période ({file_date.strftime('%Y-%m-%d')})"

    # Vérifier l'heure
    if not (TIME_START <= file_time <= TIME_END):
        return False, f"Heure hors plage ({file_time.strftime('%H:%M:%S')})"

    return True, "OK"


def get_directory_size(directory):
    """Calcule la taille totale d'un répertoire en bytes"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    total += os.path.getsize(filepath)
    except Exception as e:
        print(f"Erreur calcul taille {directory}: {e}")
    return total


def copy_single_file(file_info, target_folder):
    """
    Copie un seul fichier - fonction utilisée par le multiprocessing
    Retourne (success, file_info, error_message)
    """
    try:
        source_path = file_info["path"]
        target_path = os.path.join(target_folder, file_info["filename"])

        # Créer le répertoire cible si nécessaire
        os.makedirs(target_folder, exist_ok=True)

        # Copier le fichier avec métadonnées
        shutil.copy2(source_path, target_path)

        return True, file_info, None

    except Exception as e:
        return False, file_info, str(e)


def get_available_space(path):
    """Obtient l'espace disque disponible pour un chemin donné"""
    try:
        # Utiliser le répertoire parent s'il n'existe pas encore
        check_path = path
        while not os.path.exists(check_path):
            parent = os.path.dirname(check_path)
            if parent == check_path:  # Racine atteinte
                break
            check_path = parent

        result = subprocess.run(
            ["df", "-B1", check_path], capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                fields = lines[1].split()
                if len(fields) >= 4:
                    return int(fields[3])
    except:
        pass
    return None


def format_size(size_bytes):
    """Formate une taille en bytes vers une unité lisible"""
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}"


def analyze_source_data(source_dir):
    """
    Analyse les données sources et identifie les fichiers à copier
    """
    print("Analyse des données sources...")
    print("=" * 80)

    selected_files = {}
    total_selected_size = 0
    stats = {"total_files": 0, "selected_files": 0, "by_site": {}, "by_reason": {}}

    # Parcourir tous les dossiers
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if not os.path.isdir(item_path):
            continue

        # Extraire le serial number du nom du dossier
        serial_match = re.match(r"^([^_]+)", item)
        if not serial_match:
            continue

        serial_number = serial_match.group(1)

        print(f"\nAnalyse du dossier: {item}")
        print(f"Serial: {serial_number}")

        if serial_number in SITES_CONFIG:
            site_info = SITES_CONFIG[serial_number]
            print(f"Site cible: {site_info['target']} ({site_info['location']})")
        else:
            print("Site non sélectionné")
            continue

        # Analyser les fichiers dans ce dossier
        selected_files[item] = []
        site_stats = {"total": 0, "selected": 0, "size": 0}

        for filename in os.listdir(item_path):
            if not filename.endswith(".wav"):
                continue

            stats["total_files"] += 1
            site_stats["total"] += 1

            filepath = os.path.join(item_path, filename)
            is_selected, reason = check_file_criteria(filename, serial_number)

            if is_selected:
                file_size = os.path.getsize(filepath)
                selected_files[item].append(
                    {"filename": filename, "size": file_size, "path": filepath}
                )
                stats["selected_files"] += 1
                site_stats["selected"] += 1
                site_stats["size"] += file_size
                total_selected_size += file_size
            else:
                # Compter les raisons de rejet
                if reason not in stats["by_reason"]:
                    stats["by_reason"][reason] = 0
                stats["by_reason"][reason] += 1

        stats["by_site"][item] = site_stats
        print(f"  Fichiers trouvés: {site_stats['total']}")
        print(f"  Fichiers sélectionnés: {site_stats['selected']}")
        print(f"  Taille sélectionnée: {format_size(site_stats['size'])}")

    return selected_files, total_selected_size, stats


def print_summary(stats, total_size):
    """Affiche un résumé de l'analyse"""
    print("\n" + "=" * 80)
    print("RÉSUMÉ DE L'ANALYSE")
    print("=" * 80)

    print(f"Fichiers totaux analysés: {stats['total_files']:,}")
    print(f"Fichiers sélectionnés: {stats['selected_files']:,}")
    print(f"Taille totale à copier: {format_size(total_size)}")
    print(
        f"Pourcentage sélectionné: {(stats['selected_files']/max(stats['total_files'],1)*100):.1f}%"
    )

    print(f"\nDétail par site:")
    for site, site_stats in stats["by_site"].items():
        if site_stats["selected"] > 0:
            serial = re.match(r"^([^_]+)", site).group(1)
            site_info = SITES_CONFIG.get(serial, {})
            target_name = site_info.get("target", "Inconnu")
            print(f"  {site} ({target_name}):")
            print(
                f"    Sélectionnés: {site_stats['selected']:,} / {site_stats['total']:,}"
            )
            print(f"    Taille: {format_size(site_stats['size'])}")

    print(f"\nRaisons de rejet:")
    for reason, count in stats["by_reason"].items():
        print(f"  {reason}: {count:,} fichiers")


def copy_selected_files(source_dir, selected_files, target_dir):
    """
    Copie les fichiers sélectionnés vers le répertoire cible en utilisant le multiprocessing
    """
    print(f"\nCopie vers: {target_dir}")
    print("=" * 80)

    # Créer le répertoire cible
    os.makedirs(target_dir, exist_ok=True)

    # Déterminer le nombre de processus (CPU count - 1, minimum 1)
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Utilisation de {num_processes} processus pour la copie")

    total_copied = 0
    total_size_copied = 0
    total_errors = 0

    # Préparer toutes les tâches de copie
    copy_tasks = []
    for source_folder, files in selected_files.items():
        if not files:
            continue

        target_folder = os.path.join(target_dir, source_folder)
        print(f"\nPréparation de {source_folder}: {len(files)} fichiers")

        for file_info in files:
            copy_tasks.append((file_info, target_folder))

    if not copy_tasks:
        print("Aucun fichier à copier.")
        return 0, 0

    print(f"\nDémarrage de la copie de {len(copy_tasks)} fichiers...")
    print("=" * 80)

    # Chronométrer la copie
    start_time = time_module.time()

    # Utiliser ProcessPoolExecutor pour la copie parallèle
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Soumettre toutes les tâches
        future_to_task = {
            executor.submit(copy_single_file, file_info, target_folder): (
                file_info,
                target_folder,
            )
            for file_info, target_folder in copy_tasks
        }

        # Traiter les résultats au fur et à mesure
        for future in as_completed(future_to_task):
            success, file_info, error_message = future.result()

            if success:
                total_copied += 1
                total_size_copied += file_info["size"]

                # Afficher le progrès tous les 500 fichiers
                if total_copied % 500 == 0:
                    elapsed_time = time_module.time() - start_time
                    speed_mbps = (total_size_copied / (1024 * 1024)) / max(
                        elapsed_time, 1
                    )
                    percentage = (total_copied / len(copy_tasks)) * 100
                    eta_seconds = (elapsed_time / total_copied) * (
                        len(copy_tasks) - total_copied
                    )
                    eta_minutes = eta_seconds / 60

                    print(
                        f"    Progrès: {total_copied:,}/{len(copy_tasks):,} fichiers ({percentage:.1f}%)"
                    )
                    print(
                        f"    Copiés: {format_size(total_size_copied)} - Vitesse: {speed_mbps:.1f} MB/s - ETA: {eta_minutes:.1f} min"
                    )
            else:
                total_errors += 1
                print(f"    Erreur copie {file_info['filename']}: {error_message}")

    # Statistiques finales
    end_time = time_module.time()
    total_time = end_time - start_time
    avg_speed_mbps = (total_size_copied / (1024 * 1024)) / max(total_time, 1)

    print(f"\n✅ Copie terminée!")
    print(f"  Fichiers copiés: {total_copied:,}")
    print(f"  Taille copiée: {format_size(total_size_copied)}")
    print(f"  Temps total: {total_time:.1f} secondes ({total_time/60:.1f} minutes)")
    print(f"  Vitesse moyenne: {avg_speed_mbps:.1f} MB/s")
    if total_errors > 0:
        print(f"  ⚠️ Erreurs: {total_errors}")

    return total_copied, total_size_copied


def main():
    """Fonction principale"""
    source_dir = "/mnt/BACK UP/ACOUSTIQUE"

    # Proposer un répertoire de destination par défaut dans le dossier parent
    source_parent = os.path.dirname(source_dir)
    default_target_dir = os.path.join(source_parent, "select_data")

    print("SÉLECTION DE DONNÉES ACOUSTIQUES")
    print("=" * 80)
    print("Critères de sélection:")
    print("- Sites: Loriaz 1600/2100, Peclerey 1400, Diosaz")
    print("- Période: 1er avril → 20 juin 2025")
    print("- Heures: 8h00 → 21h00")
    print()

    # Vérifier que le répertoire source existe
    if not os.path.exists(source_dir):
        print(f"ERREUR: Répertoire source non trouvé: {source_dir}")
        return

    # Analyser les données
    selected_files, total_size, stats = analyze_source_data(source_dir)

    # Afficher le résumé
    print_summary(stats, total_size)

    # Si aucun fichier sélectionné, arrêter
    if stats["selected_files"] == 0:
        print("\nAucun fichier ne correspond aux critères.")
        return

    # Demander le répertoire de destination
    print(f"\n" + "=" * 80)
    print("CHOIX DU RÉPERTOIRE DE DESTINATION")
    print("=" * 80)
    print(f"Répertoire par défaut proposé: {default_target_dir}")
    print(f"Espace libre estimé requis: {format_size(total_size)}")
    print()

    use_default = (
        input(f"Utiliser le répertoire par défaut ? (oui/non): ").strip().lower()
    )

    if use_default in ["oui", "o", "yes", "y"]:
        target_dir = default_target_dir
    else:
        while True:
            target_dir = input(
                "Entrez le chemin du répertoire de destination: "
            ).strip()
            if target_dir:
                target_dir = os.path.expanduser(target_dir)  # Gérer ~
                target_dir = os.path.abspath(target_dir)  # Chemin absolu

                # Vérifier que le répertoire parent existe ou peut être créé
                parent_dir = os.path.dirname(target_dir)
                if os.path.exists(parent_dir) or parent_dir == target_dir:
                    break
                else:
                    print(f"ERREUR: Le répertoire parent {parent_dir} n'existe pas.")
                    continue
            else:
                print("Veuillez entrer un chemin valide.")

    print(f"\nRépertoire de destination choisi: {target_dir}")

    # Demander confirmation à l'utilisateur
    print(f"\n" + "=" * 80)
    print("CONFIRMATION DE COPIE")
    print("=" * 80)
    print(f"Répertoire source: {source_dir}")
    print(f"Répertoire cible: {target_dir}")
    print(f"Fichiers à copier: {stats['selected_files']:,}")
    print(f"Taille totale: {format_size(total_size)}")

    # Vérifier l'espace disque disponible sur le répertoire de destination
    available_space = get_available_space(target_dir)
    if available_space is not None:
        print(f"Espace disponible: {format_size(available_space)}")

        if total_size > available_space:
            print("⚠️  ATTENTION: Espace disque insuffisant!")
        elif total_size > available_space * 0.9:
            print("⚠️  ATTENTION: L'espace disque sera presque plein!")
    else:
        print("⚠️  Impossible de vérifier l'espace disque disponible.")

    print()
    response = input("Voulez-vous procéder à la copie ? (oui/non): ").strip().lower()

    if response in ["oui", "o", "yes", "y"]:
        print("\nDémarrage de la copie...")

        # Vérifier si le répertoire cible existe déjà
        if os.path.exists(target_dir):
            print(f"⚠️  Le répertoire {target_dir} existe déjà.")
            overwrite = input("Voulez-vous le remplacer ? (oui/non): ").strip().lower()
            if overwrite in ["oui", "o", "yes", "y"]:
                shutil.rmtree(target_dir)
                print("Répertoire existant supprimé.")
            else:
                print("Copie annulée.")
                return

        # Effectuer la copie
        copied_count, copied_size = copy_selected_files(
            source_dir, selected_files, target_dir
        )

        print(f"\n✅ Copie terminée avec succès!")
        print(f"   Répertoire: {target_dir}")
        print(f"   Fichiers: {copied_count:,}")
        print(f"   Taille: {format_size(copied_size)}")

    else:
        print("Copie annulée par l'utilisateur.")


if __name__ == "__main__":
    # Protection nécessaire pour le multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
