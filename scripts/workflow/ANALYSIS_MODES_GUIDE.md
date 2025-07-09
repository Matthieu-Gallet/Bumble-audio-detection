# Analysis Modes Guide

## Overview

The `final_batch_process.py` script now supports three different analysis modes to give you maximum flexibility in evaluating your acoustic detection data.

## Analysis Modes

### 1. Individual Session Analysis (`--analysis-mode individual`)

**What it does:**
- Analyzes each recording session separately
- Creates individual evaluation reports for each session
- Compares performance across sessions
- Identifies session-specific patterns and issues

**Benefits:**
- Understand how performance varies between different recording sessions
- Identify temporal trends or seasonal effects
- Detect session-specific problems (e.g., equipment issues, environmental factors)
- Compare performance across different locations or conditions

**Output Structure:**
```
output_batch/
├── eval_individual_[session]/
│   ├── evaluation_summary_[session].csv
│   ├── optimal_evaluation_summary_[session].csv
│   ├── eval_[column]/
│   └── threshold_comparison.png
└── cross_session_analysis/
    ├── all_sessions_standard.csv
    ├── all_sessions_optimal.csv
    ├── cross_session_comparison.png
    └── column_performance_summary.csv
```

**Use Cases:**
- Quality control: Identify sessions with poor performance
- Temporal analysis: Track performance changes over time
- Comparative studies: Compare different recording conditions
- Troubleshooting: Find session-specific issues

### 2. Combined Analysis (`--analysis-mode combined`)

**What it does:**
- Analyzes all sessions together as one large dataset
- Traditional approach for overall model evaluation
- Provides global performance metrics
- Optimal for overall model assessment

**Benefits:**
- Maximum statistical power with all data combined
- Overall model performance assessment
- Standard evaluation approach for model comparison
- Robust metrics with larger sample sizes

**Output Structure:**
```
output_batch/
├── eval_[column]/
│   ├── optimal_[column]/
│   ├── segments.csv
│   ├── evaluation_metrics.json
│   └── error_analysis_detailed_[column].png
├── evaluation_summary.csv
├── optimal_evaluation_summary.csv
└── threshold_comparison.png
```

**Use Cases:**
- Model validation: Overall performance assessment
- Publication: Standard evaluation metrics
- Comparison: Compare different models or approaches
- Benchmarking: Establish baseline performance

### 3. Both Modes (`--analysis-mode both`) - **DEFAULT**

**What it does:**
- Runs both individual and combined analysis
- Provides comprehensive insights at all levels
- Maximum analytical power

**Benefits:**
- Complete picture: Both session-specific and overall insights
- Comprehensive reporting: All possible analyses
- Flexibility: Can answer both detailed and broad questions
- Best value: Complete analysis in one run

## Command Examples

### Basic Usage (Both modes - Default)
```bash
python final_batch_process.py
```

### Individual Session Analysis Only
```bash
python final_batch_process.py --analysis-mode individual
```

### Combined Analysis Only
```bash
python final_batch_process.py --analysis-mode combined
```

### Analyze Specific Sessions
```bash
python final_batch_process.py --sessions 20240408_session_01_Tent_SM05_T 20240612_session_02_Tent_SM06_T
```

### Skip Data Processing (Evaluation Only)
```bash
python final_batch_process.py --skip-processing
```

### Evaluate Specific Columns
```bash
python final_batch_process.py --columns tag_Buzz buzz
```

### Combined Example
```bash
python final_batch_process.py --analysis-mode individual --sessions 20240408_session_01_Tent_SM05_T --columns tag_Buzz tag_Insect
```

## Key Features

### 🎯 **Session-Specific Analysis**
- Performance metrics per session
- Error analysis by session
- Optimal thresholds per session
- Session comparison heatmaps

### 📊 **Cross-Session Comparison**
- Performance trends across sessions
- Statistical summaries by column
- Variance analysis between sessions
- Visualization of session differences

### 🔍 **Flexible Target Selection**
- Choose specific sessions to analyze
- Select specific columns to evaluate
- Skip processing for evaluation-only runs
- Combine multiple options

### 📈 **Advanced Analytics**
- Optimal threshold detection per session
- False positive/negative analysis per session
- Class-based error breakdown
- Performance improvement quantification

## Recommendations

### For Quality Control
```bash
python final_batch_process.py --analysis-mode individual
```
Look at cross-session comparison to identify problematic sessions.

### For Model Evaluation
```bash
python final_batch_process.py --analysis-mode combined
```
Use combined analysis for robust overall performance metrics.

### For Comprehensive Research
```bash
python final_batch_process.py --analysis-mode both
```
Get complete insights at all levels (default behavior).

### For Troubleshooting Specific Sessions
```bash
python final_batch_process.py --analysis-mode individual --sessions [problematic_session]
```

### For Quick Re-evaluation
```bash
python final_batch_process.py --skip-processing --analysis-mode combined
```
Skip data processing and go directly to evaluation.

## Output Interpretation

### Individual Mode Results
- **Session Performance**: Check `eval_individual_[session]/` for each session
- **Cross-Session Trends**: See `cross_session_analysis/cross_session_comparison.png`
- **Performance Variance**: Check `column_performance_summary.csv` for statistics

### Combined Mode Results
- **Overall Performance**: See `evaluation_summary.csv` and `optimal_evaluation_summary.csv`
- **Detailed Analysis**: Check `eval_[column]/optimal_[column]/` directories
- **Error Analysis**: Review detailed error tables and visualizations

This flexible approach allows you to choose the right level of analysis for your specific needs while maintaining the ability to get comprehensive insights when needed.
