#!/usr/bin/env python3
"""
Configuration loader for the workflow system.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging


class WorkflowConfig:
    """Configuration management for the workflow system."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_file: Path to configuration file. If None, uses default.
        """
        self.project_root = self._find_project_root()

        if config_file is None:
            config_file = os.path.join(
                self.project_root, "scripts", "config", "workflow_config.yaml"
            )

        self.config_file = config_file
        self.config = self._load_config()
        self._resolve_paths()

    def _find_project_root(self) -> str:
        """Find the project root directory."""
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Go up directories until we find the project root
        # (look for key files like process.py, requirements.txt)
        search_dir = current_dir
        for _ in range(5):  # Max 5 levels up
            if os.path.exists(
                os.path.join(search_dir, "process.py")
            ) and os.path.exists(os.path.join(search_dir, "requirements.txt")):
                return search_dir
            search_dir = os.path.dirname(search_dir)

        # Fallback: assume we're in scripts/workflow and project root is 2 levels up
        return os.path.dirname(os.path.dirname(current_dir))

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, "r") as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"⚠️  Configuration file not found: {self.config_file}")
            print("📝 Using default configuration")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"❌ Error parsing configuration file: {e}")
            print("📝 Using default configuration")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when file is not available."""
        return {
            "paths": {
                "data_dir": "data",
                "python_path": ".venv/bin/python",
                "process_script": "process.py",
                "output_dir": "output_batch",
            },
            "processing": {"segment_length": 5, "audio_format": "wav", "timeout": 1800},
            "analysis": {
                "mode": "both",
                "columns": ["tag_Buzz", "tag_Insect", "biophony"],
                "default_threshold": 0.5,
                "evaluation_duration": 5.0,
            },
            "sessions": {"target_sessions": [], "skip_processing": False},
            "evaluation": {
                "features": {
                    "optimal_threshold": True,
                    "error_analysis": True,
                    "class_breakdown": True,
                    "visualizations": True,
                    "cross_session_comparison": True,
                }
            },
        }

    def _resolve_paths(self):
        """Resolve all paths to absolute paths."""
        paths = self.config.get("paths", {})

        # Convert relative paths to absolute paths
        for key, path in paths.items():
            if not os.path.isabs(path):
                paths[key] = os.path.join(self.project_root, path)

    # Path properties
    @property
    def data_path(self) -> str:
        """Get data directory path."""
        return self.config["paths"]["data_dir"]

    @property
    def python_path(self) -> str:
        """Get Python executable path."""
        return self.config["paths"]["python_path"]

    @property
    def process_script(self) -> str:
        """Get process script path."""
        return self.config["paths"]["process_script"]

    @property
    def output_base(self) -> str:
        """Get output base directory."""
        return self.config["paths"]["output_dir"]

    # Processing properties
    @property
    def segment_length(self) -> int:
        """Get segment length."""
        return self.config["processing"]["segment_length"]

    @property
    def audio_format(self) -> str:
        """Get audio format."""
        return self.config["processing"]["audio_format"]

    @property
    def timeout(self) -> int:
        """Get processing timeout."""
        return self.config["processing"]["timeout"]

    # Analysis properties
    @property
    def analysis_mode(self) -> str:
        """Get analysis mode."""
        return self.config["analysis"]["mode"]

    @property
    def columns(self) -> List[str]:
        """Get columns to analyze."""
        return self.config["analysis"]["columns"]

    @property
    def default_threshold(self) -> float:
        """Get default threshold."""
        return self.config["analysis"]["default_threshold"]

    @property
    def evaluation_duration(self) -> float:
        """Get evaluation duration."""
        return self.config["analysis"]["evaluation_duration"]

    # Session properties
    @property
    def target_sessions(self) -> List[str]:
        """Get target sessions."""
        return self.config["sessions"]["target_sessions"]

    @property
    def skip_processing(self) -> bool:
        """Get skip processing flag."""
        return self.config["sessions"]["skip_processing"]

    # Feature flags
    @property
    def features(self) -> Dict[str, bool]:
        """Get enabled features."""
        return self.config.get("evaluation", {}).get("features", {})

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path (e.g., 'paths.data_dir')."""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value by key path."""
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, config_file: Optional[str] = None):
        """Save configuration to file."""
        if config_file is None:
            config_file = self.config_file

        with open(config_file, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

    def print_config(self):
        """Print current configuration."""
        print("📋 CURRENT CONFIGURATION:")
        print("=" * 50)
        print(f"📁 Project root: {self.project_root}")
        print(f"📄 Config file: {self.config_file}")
        print()

        print("🗂️  PATHS:")
        print(f"  Data: {self.data_path}")
        print(f"  Python: {self.python_path}")
        print(f"  Process script: {self.process_script}")
        print(f"  Output: {self.output_base}")
        print()

        print("⚙️  PROCESSING:")
        print(f"  Segment length: {self.segment_length}s")
        print(f"  Audio format: {self.audio_format}")
        print(f"  Timeout: {self.timeout}s")
        print()

        print("📊 ANALYSIS:")
        print(f"  Mode: {self.analysis_mode}")
        print(f"  Columns: {', '.join(self.columns)}")
        print(f"  Default threshold: {self.default_threshold}")
        print()

        if self.target_sessions:
            print("🎯 TARGET SESSIONS:")
            for session in self.target_sessions:
                print(f"  - {session}")
        else:
            print("🎯 TARGET SESSIONS: All available")

        print()
        print(f"⏩ Skip processing: {self.skip_processing}")


def load_config(config_file: Optional[str] = None) -> WorkflowConfig:
    """Load workflow configuration."""
    return WorkflowConfig(config_file)


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    config.print_config()
