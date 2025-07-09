# Audio Detection Automation System

A comprehensive automated audio detection and evaluation system for soundscape analysis, specifically designed for the BumbleBuzz project. This system provides end-to-end automation from audio processing to performance evaluation with advanced error analysis.

## 🎯 Overview

The Audio Detection Automation System automates the complete workflow of:
1. **Audio Processing**: Batch processing of audio files using PANNs (Pretrained Audio Neural Networks)
2. **Detection Analysis**: Multi-column detection analysis with optimal threshold finding
3. **Performance Evaluation**: Comprehensive evaluation with error analysis and visualizations
4. **Workflow Management**: Flexible configuration system and monitoring tools

## 📁 Package Structure

```
detection/
├── main.py                     # Main launcher with interactive menu
├── process.py                  # Core audio processing script
├── setup.py                    # Environment setup and validation
├── requirements.txt            # Python dependencies
├── dash_app.py                # Web dashboard for results visualization
├── models.py                   # Audio model definitions
├── indices.py                  # Audio indices calculations
├── 
├── data/                       # Audio data and annotations
│   ├── 20240408_session_01_Tent/
│   ├── 20240612_session_02_Tent/
│   └── ...
├── 
├── scripts/                    # Main automation scripts
│   ├── config/                 # Configuration system
│   │   ├── config_loader.py    # Configuration loader
│   │   ├── workflow_config.yaml # Main configuration file
│   │   ├── example_config.yaml # Example configuration
│   │   └── README.md          # Configuration documentation
│   │
│   ├── workflow/               # Main workflow scripts
│   │   ├── final_batch_process.py  # Complete automation workflow
│   │   └── run_full_automation.py  # Interactive workflow launcher
│   │
│   ├── evaluation/             # Evaluation and analysis
│   │   ├── evaluate_detection.py   # Detection evaluation engine
│   │   └── advanced_evaluation.py # Advanced analysis tools
│   │
│   ├── utilities/              # Utility scripts
│   │   ├── check_results.py    # Results summary and overview
│   │   ├── monitor_workflow.py # Real-time workflow monitoring
│   │   ├── view_docs.py        # Interactive documentation
│   │   ├── config_demo.py      # Configuration examples
│   │   ├── clean_results.py    # Results cleanup
│   │   └── summary_improvements.py # Performance summaries
│   │
│   └── testing/                # Testing and validation
│       ├── test_full_workflow.py   # Workflow testing
│       ├── demo_error_analysis.py # Error analysis demo
│       └── test_discovery.py      # System discovery tests
│
├── utils/                      # Core utilities
│   ├── alpha_indices.py        # Audio indices calculations
│   ├── config.py              # System configuration
│   ├── dataloader.py          # Data loading utilities
│   ├── ecoacoustics.py        # Ecoacoustic analysis
│   ├── metadata.py            # Metadata handling
│   ├── pytorch_utils.py       # PyTorch utilities
│   ├── stft.py                # STFT analysis
│   └── utils.py               # General utilities
│
├── output_batch/              # Processing results
│   ├── merged_results.csv     # Combined results
│   ├── evaluation_summary.csv # Performance summary
│   ├── [session_name]/        # Individual session results
│   └── eval_[column]/         # Evaluation results per column
│
└── assets/                    # Test files and resources
    ├── test.flac
    ├── tps1.flac
    └── tps2.flac
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run environment check
python setup.py

# Activate virtual environment (if using)
source .venv/bin/activate
```

### 2. Basic Usage
```bash
# Launch interactive system
python main.py

# Or run directly
python scripts/workflow/final_batch_process.py
```

### 3. Configuration
```bash
# Edit configuration
nano scripts/config/workflow_config.yaml

# Print current configuration
python scripts/workflow/final_batch_process.py --print-config
```

### 4. Monitoring
```bash
# Check results
python scripts/utilities/check_results.py

# Monitor in real-time
python scripts/utilities/monitor_workflow.py --watch
```

## ⚙️ Configuration System

The system uses a flexible YAML-based configuration system located in `scripts/config/`:

### Configuration Files
- `workflow_config.yaml`: Main configuration file
- `example_config.yaml`: Example configuration templates
- `config_loader.py`: Configuration loading system

### Key Configuration Options

#### Paths
```yaml
paths:
  data_dir: "data"                    # Audio data directory
  output_dir: "output_batch"          # Results output directory
  python_path: ".venv/bin/python"     # Python executable
  process_script: "process.py"        # Main processing script
```

#### Analysis Settings
```yaml
analysis:
  mode: "both"                        # individual, combined, or both
  columns: ["tag_Buzz", "tag_Insect"] # Detection columns to analyze
  default_threshold: 0.5              # Default detection threshold
```

#### Processing Parameters
```yaml
processing:
  segment_length: 5                   # Audio segment length (seconds)
  audio_format: "wav"                 # Audio file format
  timeout: 1800                       # Processing timeout (seconds)
```

## 🔧 Core Functions

### 1. Audio Processing (`process.py`)
```bash
python process.py --data_path /path/to/audio --save_path /path/to/output --name session_name
```

**Functions:**
- Audio file discovery and validation
- PANNs model inference for audio tagging
- Feature extraction and indexing
- Multi-format audio support (WAV, FLAC)

### 2. Batch Processing (`scripts/workflow/final_batch_process.py`)
```bash
python scripts/workflow/final_batch_process.py [OPTIONS]
```

**Options:**
- `--analysis-mode {individual,combined,both}`: Analysis mode
- `--sessions [SESSION ...]`: Specific sessions to process
- `--columns [COLUMN ...]`: Detection columns to evaluate
- `--skip-processing`: Skip processing, run evaluation only
- `--config CONFIG`: Custom configuration file

**Functions:**
- Automated directory discovery
- Parallel processing management
- Results merging and consolidation
- Comprehensive evaluation pipeline

### 3. Detection Evaluation (`scripts/evaluation/evaluate_detection.py`)
**Functions:**
- Ground truth loading and validation
- Prediction analysis and metrics calculation
- Optimal threshold detection
- Confusion matrix generation
- Performance visualization

### 4. Configuration Management (`scripts/config/config_loader.py`)
**Functions:**
- YAML configuration loading
- Path resolution and validation
- Default value handling
- Environment-specific configuration

## 📊 Analysis Modes

### Individual Analysis
Analyzes each session separately, providing:
- Session-specific performance metrics
- Cross-session comparison
- Individual error analysis
- Session-level visualizations

### Combined Analysis
Analyzes all sessions together, providing:
- Overall system performance
- Aggregated metrics
- Combined error analysis
- System-wide visualizations

### Both Analysis
Runs both individual and combined analysis, providing:
- Complete performance overview
- Detailed session comparisons
- Comprehensive error analysis
- Full visualization suite

## 🔍 Monitoring and Diagnostics

### Real-time Monitoring
```bash
python scripts/utilities/monitor_workflow.py --watch
```

**Features:**
- Live progress tracking
- Session completion status
- Error detection and reporting
- Performance metrics display

### Results Checking
```bash
python scripts/utilities/check_results.py
```

**Features:**
- Results summary and overview
- File existence validation
- Performance metrics display
- Error analysis summary

### System Diagnostics
```bash
python setup.py
```

**Features:**
- Environment validation
- Dependency checking
- Configuration validation
- System health checks

## 🧪 Testing and Validation

### Workflow Testing
```bash
python scripts/testing/test_full_workflow.py
```

### Error Analysis Demo
```bash
python scripts/testing/demo_error_analysis.py
```

### Configuration Testing
```bash
python scripts/utilities/config_demo.py
```

## 📈 Performance Evaluation

### Standard Metrics
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Accuracy**: (True positives + True negatives) / Total samples

### Advanced Analysis
- **Optimal Threshold Detection**: Finds threshold that maximizes F1-score
- **Error Analysis**: Detailed false positive/negative analysis
- **Class-based Breakdown**: Performance by audio class
- **Cross-session Comparison**: Performance across different sessions

### Visualization
- Precision-Recall curves
- ROC curves
- Confusion matrices
- Error distribution plots
- Performance comparison charts

## 🎛️ Web Dashboard

```bash
python dash_app.py
```

**Features:**
- Interactive results visualization
- Real-time performance monitoring
- Audio playback and analysis
- Annotation tools
- Export capabilities

## 📋 Data Requirements

### Audio Files
- **Format**: WAV or FLAC
- **Naming**: YYMMDD_HHMMSS.wav or prefix_YYMMDD_HHMMSS.wav
- **Organization**: Grouped by session folders

### Annotation Files
- **Format**: Text files (.txt)
- **Structure**: Time-based annotations
- **Organization**: Parallel annotation folders (session_annotées)

### Directory Structure
```
data/
├── 20240408_session_01_Tent/
│   ├── SM05_T/                 # Audio files
│   │   ├── file1.wav
│   │   └── file2.wav
│   └── SM05_T_annotées/        # Annotations
│       ├── file1.txt
│       └── file2.txt
```

## 🛠️ Development and Extension

### Adding New Detection Columns
1. Update configuration file with new column names
2. Modify evaluation scripts to handle new columns
3. Update visualization components

### Custom Analysis Modes
1. Extend `final_batch_process.py` with new mode
2. Add configuration options
3. Update command-line interface

### New Evaluation Metrics
1. Extend `evaluate_detection.py` with new metrics
2. Update visualization components
3. Add to summary reports

## 📚 Documentation

### Configuration Guide
See `scripts/config/README.md` for detailed configuration instructions.

### API Documentation
All functions include comprehensive docstrings with parameter descriptions and examples.

### Usage Examples
Use the interactive documentation viewer:
```bash
python scripts/utilities/view_docs.py
```

## 🔧 Troubleshooting

### Common Issues

1. **Configuration Not Found**
   - Check config file path: `scripts/config/workflow_config.yaml`
   - Validate YAML syntax
   - Ensure all required fields are present

2. **Processing Errors**
   - Check audio file format and structure
   - Verify annotation file alignment
   - Monitor system resources

3. **Import Errors**
   - Verify virtual environment activation
   - Check Python path configuration
   - Install missing dependencies

### Debug Mode
```bash
python scripts/workflow/final_batch_process.py --print-config
```

### Log Files
Check `output_batch/workflow_log.txt` for detailed execution logs.

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request


## 📊 Performance Metrics

The system provides comprehensive performance analysis including:
- **Processing Speed**: Audio processing rate and throughput
- **Detection Accuracy**: Multi-class detection performance
- **System Resources**: Memory and CPU usage monitoring
- **Error Analysis**: Detailed error classification and analysis

## 🔄 Workflow Summary

1. **Setup**: Environment validation and configuration
2. **Discovery**: Automatic audio and annotation file discovery
3. **Processing**: Batch audio processing with PANNs
4. **Merging**: Results consolidation and validation
5. **Evaluation**: Comprehensive performance evaluation
6. **Analysis**: Advanced error analysis and visualization
7. **Reporting**: Summary generation and export

This system provides a complete solution for automated audio detection with professional-grade evaluation and monitoring capabilities.
