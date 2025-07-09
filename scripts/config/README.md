# Configuration System Guide

## Overview

The workflow now supports a flexible configuration system that allows you to customize all aspects of the analysis without modifying the code.

## Configuration Files

### Main Configuration File
- **Location**: `config/workflow_config.yaml`
- **Purpose**: Default configuration for all settings
- **Format**: YAML

### Example Configuration File
- **Location**: `config/example_config.yaml`
- **Purpose**: Simplified configuration template for common use cases
- **Format**: YAML

## Quick Start

### 1. Basic Usage (Default Configuration)
```bash
python final_batch_process.py
```

### 2. Using Custom Configuration
```bash
python final_batch_process.py --config path/to/your/config.yaml
```

### 3. Print Current Configuration
```bash
python final_batch_process.py --print-config
```

### 4. Override Configuration via Command Line
```bash
python final_batch_process.py --analysis-mode individual --columns tag_Buzz buzz
```

## Configuration Options

### Analysis Settings
```yaml
analysis:
  mode: "both"                    # individual, combined, or both
  columns:                        # Columns to evaluate
    - "tag_Buzz"
    - "tag_Insect"
    - "buzz"
  default_threshold: 0.5          # Default threshold for evaluation
  evaluation_duration: 5.0        # Duration in seconds
```

### Session Selection
```yaml
sessions:
  target_sessions: []             # Empty = all sessions
  # target_sessions:              # Specific sessions
  #   - "20240408_session_01_Tent_SM05_T"
  #   - "20240612_session_02_Tent_SM06_T"
  skip_processing: false          # Skip data processing
```

### Processing Settings
```yaml
processing:
  segment_length: 5               # Audio segment length in seconds
  audio_format: "wav"             # Audio file format
  timeout: 1800                   # Processing timeout in seconds
```

### Path Configuration
```yaml
paths:
  data_dir: "data"                # Data directory
  python_path: ".venv/bin/python" # Python executable
  process_script: "process.py"    # Main processing script
  output_dir: "output_batch"      # Output directory
```

## Usage Examples

### Example 1: Analyze Only Buzz Detection
```yaml
# config/buzz_only.yaml
analysis:
  mode: "both"
  columns:
    - "tag_Buzz"
    - "buzz"

sessions:
  target_sessions: []
  skip_processing: false
```

```bash
python final_batch_process.py --config config/buzz_only.yaml
```

### Example 2: Re-run Evaluation Only
```yaml
# config/eval_only.yaml
analysis:
  mode: "combined"
  columns:
    - "tag_Buzz"
    - "tag_Insect"
    - "buzz"

sessions:
  target_sessions: []
  skip_processing: true  # Skip data processing
```

```bash
python final_batch_process.py --config config/eval_only.yaml
```

### Example 3: Analyze Specific Sessions
```yaml
# config/specific_sessions.yaml
analysis:
  mode: "individual"
  columns:
    - "tag_Buzz"
    - "buzz"

sessions:
  target_sessions:
    - "20240408_session_01_Tent_SM05_T"
    - "20240612_session_02_Tent_SM06_T"
  skip_processing: false
```

```bash
python final_batch_process.py --config config/specific_sessions.yaml
```

## Priority Order

Configuration values are applied in this order (highest to lowest priority):

1. **Command line arguments** (highest priority)
2. **Custom configuration file** (specified with --config)
3. **Default configuration file** (config/workflow_config.yaml)
4. **Built-in defaults** (lowest priority)

## Command Line Overrides

You can override any configuration setting via command line:

```bash
# Override analysis mode
python final_batch_process.py --analysis-mode individual

# Override columns
python final_batch_process.py --columns tag_Buzz buzz

# Override sessions
python final_batch_process.py --sessions 20240408_session_01_Tent_SM05_T

# Skip processing
python final_batch_process.py --skip-processing

# Combine multiple overrides
python final_batch_process.py --config my_config.yaml --analysis-mode individual --skip-processing
```

## Creating Your Own Configuration

### Method 1: Copy and Modify Example
```bash
cp config/example_config.yaml config/my_config.yaml
# Edit my_config.yaml with your settings
python final_batch_process.py --config config/my_config.yaml
```

### Method 2: Start from Scratch
```yaml
# my_custom_config.yaml
analysis:
  mode: "individual"
  columns:
    - "tag_Buzz"

sessions:
  target_sessions:
    - "20240408_session_01_Tent_SM05_T"
  skip_processing: false
```

## Troubleshooting

### Configuration Not Loading
- Check file path: `--config path/to/file.yaml`
- Verify YAML syntax: Use a YAML validator
- Check file permissions: Ensure file is readable

### Invalid Configuration Values
- Check spelling of options (case-sensitive)
- Verify data types (strings, lists, booleans)
- Ensure paths exist and are accessible

### Command Line vs Configuration Conflicts
- Command line arguments always override configuration
- Use `--print-config` to see final configuration
- Check priority order above

## Advanced Features

### Environment Variables
You can use environment variables in paths:
```yaml
paths:
  data_dir: "${HOME}/my_data"
  output_dir: "${WORK_DIR}/output"
```

### Conditional Configuration
Different configurations for different environments:
```yaml
# Development
analysis:
  mode: "individual"
  columns: ["tag_Buzz"]

# Production
# analysis:
#   mode: "both"
#   columns: ["tag_Buzz", "tag_Insect", "buzz"]
```

This configuration system provides maximum flexibility while maintaining ease of use!
