#!/usr/bin/env python3
"""
Simple script to test directory discovery.
"""

import os
import glob
from pathlib import Path


def find_data_directories(base_data_path: str) -> list:
    """Find all data directories."""
    print(f"Searching in: {base_data_path}")

    if not os.path.exists(base_data_path):
        print(f"Directory {base_data_path} does not exist!")
        return []

    data_dirs = []

    for item in os.listdir(base_data_path):
        item_path = os.path.join(base_data_path, item)
        print(f"  Examining: {item}")

        if os.path.isdir(item_path) and not item.endswith("_annotées"):
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                print(f"    Subdirectory: {subitem}")

                if os.path.isdir(subitem_path) and not subitem.endswith("_annotées"):
                    # Check if there are audio files
                    audio_files = glob.glob(os.path.join(subitem_path, "*.wav"))
                    print(f"      Audio files found: {len(audio_files)}")

                    if audio_files:
                        data_dirs.append(
                            {
                                "session": item,
                                "subdirectory": subitem,
                                "full_path": subitem_path,
                                "audio_count": len(audio_files),
                            }
                        )

    return data_dirs


def find_annotation_directories(base_data_path: str) -> list:
    """Find all annotation directories."""
    print(f"Searching for annotations in: {base_data_path}")

    annotation_dirs = []

    for item in os.listdir(base_data_path):
        item_path = os.path.join(base_data_path, item)

        if os.path.isdir(item_path):
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)

                if os.path.isdir(subitem_path) and subitem.endswith("_annotées"):
                    annotation_files = glob.glob(os.path.join(subitem_path, "*.txt"))
                    print(
                        f"    Annotation directory: {item}/{subitem} ({len(annotation_files)} files)"
                    )

                    if annotation_files:
                        annotation_dirs.append(
                            {
                                "session": item,
                                "subdirectory": subitem,
                                "full_path": subitem_path,
                                "annotation_count": len(annotation_files),
                            }
                        )

    return annotation_dirs


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    BASE_DATA_PATH = os.path.join(parent_dir, "data")

    print("DATA DIRECTORY ANALYSIS")
    print("=" * 50)

    # Data directories
    data_dirs = find_data_directories(BASE_DATA_PATH)
    print(f"\n{len(data_dirs)} data directories found:")
    for data_dir in data_dirs:
        print(
            f"  - {data_dir['session']}/{data_dir['subdirectory']} ({data_dir['audio_count']} files)"
        )
        print(f"    Path: {data_dir['full_path']}")

    # Annotation directories
    annotation_dirs = find_annotation_directories(BASE_DATA_PATH)
    print(f"\n{len(annotation_dirs)} annotation directories found:")
    for ann_dir in annotation_dirs:
        print(
            f"  - {ann_dir['session']}/{ann_dir['subdirectory']} ({ann_dir['annotation_count']} files)"
        )
        print(f"    Path: {ann_dir['full_path']}")
