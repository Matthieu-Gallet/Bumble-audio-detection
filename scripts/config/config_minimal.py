#!/usr/bin/env python3
"""
Very simple configuration class without external dependencies.
"""

import os
import yaml
from typing import List, Dict, Any


class WorkflowConfig:
    """Simple configuration management."""

    def __init__(self, config_file=None):
        """Initialize with default configuration."""
        self.project_root = self._find_project_root()
        self.config_file = config_file or "simple_config.yaml"

        # Default configuration
        self._data_dir = "data"
        self._python_path = ".venv/bin/python"
        self._process_script = "process.py"
        self._output_dir = "output_batch"
        self._segment_length = 5
        self._audio_format = "wav"
        self._timeout = 1800
        self._analysis_mode = "inference"
        self._columns = ["Group_buzz"]  # Évaluation uniquement pour Buzz
        self._default_threshold = (
            0.001  # Seuil très bas pour détecter les faibles probabilités
        )
        self._annotations_dir = "data"
        self._annotation_pattern = "*_annotées"
        self._results_subdir = "results"

        # Model configuration
        self._model_type = "mobilenetv2"

        # Advanced evaluation settings
        self._evaluation_enabled = True
        self._optimize_threshold = True
        self._evaluation_duration = 10.0
        self._evaluation_output_dir = "evaluation_advanced"

        # Load YAML config if provided and exists
        self._load_yaml_config()

        # Resolve paths to absolute
        self._resolve_paths()

    def _find_project_root(self):
        """Find the project root directory."""
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Go up directories until we find the project root
        search_dir = current_dir
        for _ in range(5):
            if os.path.exists(os.path.join(search_dir, "process.py")):
                return search_dir
            search_dir = os.path.dirname(search_dir)

        # Fallback
        return os.path.dirname(os.path.dirname(current_dir))

    def _load_yaml_config(self):
        """Load configuration from YAML file if it exists."""
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    yaml_config = yaml.safe_load(f)

                if yaml_config:
                    # Handle nested structure with backwards compatibility
                    if "paths" in yaml_config:
                        paths = yaml_config["paths"]
                        self._data_dir = paths.get("data_dir", self._data_dir)
                        self._output_dir = paths.get("output_dir", self._output_dir)
                        self._python_path = paths.get("python_path", self._python_path)
                        self._process_script = paths.get(
                            "process_script", self._process_script
                        )

                    if "processing" in yaml_config:
                        processing = yaml_config["processing"]
                        self._segment_length = processing.get(
                            "segment_length", self._segment_length
                        )
                        self._audio_format = processing.get(
                            "audio_format", self._audio_format
                        )
                        self._timeout = processing.get("timeout", self._timeout)

                    # Model configuration
                    if "model" in yaml_config:
                        model_config = yaml_config["model"]
                        self._model_type = model_config.get("type", self._model_type)

                    if "analysis" in yaml_config:
                        analysis = yaml_config["analysis"]
                        self._analysis_mode = analysis.get("mode", self._analysis_mode)

                        if "evaluation" in analysis:
                            eval_config = analysis["evaluation"]
                            self._columns = eval_config.get("columns", self._columns)
                            self._default_threshold = eval_config.get(
                                "default_threshold", self._default_threshold
                            )

                    # Advanced evaluation settings
                    if "advanced_evaluation" in yaml_config:
                        adv_eval = yaml_config["advanced_evaluation"]
                        self._evaluation_enabled = adv_eval.get(
                            "enabled", self._evaluation_enabled
                        )
                        self._optimize_threshold = adv_eval.get(
                            "optimize_threshold", self._optimize_threshold
                        )
                        self._evaluation_duration = adv_eval.get(
                            "duration", self._evaluation_duration
                        )
                        self._evaluation_output_dir = adv_eval.get(
                            "output_dir", self._evaluation_output_dir
                        )

                    # Backwards compatibility for flat structure
                    else:
                        self._data_dir = yaml_config.get("data_dir", self._data_dir)
                        self._output_dir = yaml_config.get(
                            "output_dir", self._output_dir
                        )
                        self._analysis_mode = yaml_config.get(
                            "analysis_mode", self._analysis_mode
                        )
                        self._columns = yaml_config.get("columns", self._columns)
                        self._default_threshold = yaml_config.get(
                            "threshold", self._default_threshold
                        )
                        self._segment_length = yaml_config.get(
                            "segment_length", self._segment_length
                        )
                        self._audio_format = yaml_config.get(
                            "audio_format", self._audio_format
                        )
                        self._timeout = yaml_config.get("timeout", self._timeout)

                        # Advanced evaluation settings from flat structure
                        eval_config = yaml_config.get("evaluation", {})
                        self._evaluation_enabled = eval_config.get(
                            "enabled", self._evaluation_enabled
                        )
                        self._optimize_threshold = eval_config.get(
                            "optimize_threshold", self._optimize_threshold
                        )
                        self._evaluation_duration = eval_config.get(
                            "duration", self._evaluation_duration
                        )
                        self._evaluation_output_dir = eval_config.get(
                            "output_dir", self._evaluation_output_dir
                        )

            except Exception as e:
                print(f"Warning: Could not load YAML config {self.config_file}: {e}")
        elif self.config_file and not os.path.exists(self.config_file):
            # Create default YAML config if it doesn't exist
            self._create_default_yaml_config()

    def _create_default_yaml_config(self):
        """Create a default YAML configuration file."""
        default_config = {
            "data_dir": self._data_dir,
            "output_dir": self._output_dir,
            "analysis_mode": self._analysis_mode,
            "columns": self._columns,
            "threshold": self._default_threshold,
            "segment_length": self._segment_length,
            "audio_format": self._audio_format,
            "timeout": self._timeout,
            "model_type": self._model_type,
            "evaluation": {
                "enabled": self._evaluation_enabled,
                "optimize_threshold": self._optimize_threshold,
                "duration": self._evaluation_duration,
                "output_dir": self._evaluation_output_dir,
            },
        }

        try:
            with open(self.config_file, "w") as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            print(f"✅ Created default config file: {self.config_file}")
        except Exception as e:
            print(f"Warning: Could not create config file {self.config_file}: {e}")

    def _resolve_paths(self):
        """Convert relative paths to absolute paths."""
        if not os.path.isabs(self._data_dir):
            self._data_dir = os.path.join(self.project_root, self._data_dir)
        if not os.path.isabs(self._python_path):
            self._python_path = os.path.join(self.project_root, self._python_path)
        if not os.path.isabs(self._process_script):
            self._process_script = os.path.join(self.project_root, self._process_script)
        if not os.path.isabs(self._output_dir):
            self._output_dir = os.path.join(self.project_root, self._output_dir)

    # Properties
    @property
    def data_path(self) -> str:
        return self._data_dir

    @property
    def python_path(self) -> str:
        return self._python_path

    @property
    def process_script(self) -> str:
        return self._process_script

    @property
    def output_base(self) -> str:
        return self._output_dir

    @property
    def segment_length(self) -> int:
        return self._segment_length

    @property
    def audio_format(self) -> str:
        return self._audio_format

    @property
    def timeout(self) -> int:
        return self._timeout

    @property
    def analysis_mode(self) -> str:
        return self._analysis_mode

    @property
    def columns(self) -> List[str]:
        return self._columns

    @property
    def default_threshold(self) -> float:
        return self._default_threshold

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def evaluation_enabled(self) -> bool:
        return self._evaluation_enabled

    @property
    def optimize_threshold(self) -> bool:
        return self._optimize_threshold

    @property
    def evaluation_duration(self) -> float:
        return self._evaluation_duration

    @property
    def advanced_evaluation(self) -> Dict[str, Any]:
        """Return advanced evaluation settings as a dictionary."""
        return {
            "enabled": self._evaluation_enabled,
            "optimize_threshold": self._optimize_threshold,
            "duration": self._evaluation_duration,
            "output_dir": self._evaluation_output_dir,
        }

    @property
    def evaluation_output_dir(self) -> str:
        return self._evaluation_output_dir

    @property
    def annotations_dir(self) -> str:
        return self._annotations_dir

    @property
    def annotation_pattern(self) -> str:
        return self._annotation_pattern

    @property
    def results_subdir(self) -> str:
        return self._results_subdir

    @property
    def is_evaluation_mode(self) -> bool:
        return self._analysis_mode == "evaluation"

    # Configuration setters for runtime modification
    def set_mode(self, mode: str):
        """Set analysis mode."""
        if mode in ["inference", "evaluation"]:
            self._analysis_mode = mode

    def set_columns(self, columns: List[str]):
        """Set columns to evaluate."""
        self._columns = columns

    def print_config(self):
        """Print current configuration."""
        print("CONFIGURATION")
        print("=" * 40)
        print(f"Mode: {self.analysis_mode}")
        print(f"Data path: {self.data_path}")
        print(f"Output path: {self.output_base}")
        print(f"Columns: {self.columns}")
        if self.is_evaluation_mode:
            print(f"Results subdir: {self.results_subdir}")


def load_config(config_file=None):
    """
    Load configuration for the evaluation scripts.

    Returns:
        dict: Configuration dictionary with all necessary parameters
    """
    config = WorkflowConfig(config_file)

    return {
        "csv_file": os.path.join(config.output_base, "merged_results.csv"),
        "annotations_dir": os.path.join(config.data_path, "*", "*_annotées"),
        "columns": config.columns,
        "threshold": config.default_threshold,
        "output_dir": os.path.join(config.output_base, config.evaluation_output_dir),
        "segment_duration": config.evaluation_duration,
        "project_root": config.project_root,
        "evaluation_enabled": config.evaluation_enabled,
        "optimize_threshold": config.optimize_threshold,
        "analysis_mode": config.analysis_mode,
        "audio_format": config.audio_format,
        "timeout": config.timeout,
    }


if __name__ == "__main__":
    config = WorkflowConfig()
    config.print_config()
