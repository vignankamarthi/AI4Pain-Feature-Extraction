# AI4PAIN 2025 Feature Extraction

A feature extraction tool for the AI4PAIN 2025 Challenge that analyzes physiological signals (EDA, BVP, RESP, SpO₂) for pain assessment. The tool calculates permutation entropy, complexity, and Fisher complexity features for signals under different pain states.

## Setup and Usage

This project uses standard Python package management with a simple manual setup:

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the feature extraction
python main.py
```

Additional options:

```bash
# To include Fisher PE verification columns in output
python main.py --verify

# To use custom embedding dimensions (default is 1,2)
python main.py --dimensions 3,4,5

# To use custom time delays (default is 1,2,3)
python main.py --taus 1,3,5

# Combining options
python main.py --verify --dimensions 3,4 --taus 2,4
```

## Visualizing Results

After generating features, you can visualize and analyze the results using:

```bash
# Run the visualization with default settings
python view_results.py

# Specify a custom feature file path
python view_results.py --file path/to/features.csv
```

The visualization script generates several outputs:

1. **DataFrame Preview**: Displays the extracted features in a tabular format, showing the actual feature values.

2. **Complexity-Entropy Causality Plane** (`results/complexity_entropy_plane.png`):
   - Plots permutation entropy vs. complexity for different signal types and pain states
   - Each subplot represents a different dimension-tau combination
   - Shows how signals distribute across the complexity-entropy space

3. **State Transitions Analysis** (`results/state_transitions.png`):
   - Shows changes in PE, complexity, and Fisher information between pain states
   - Helps identify which metrics change most significantly with pain intensity
   - Bars indicate magnitude and direction of changes between states

4. **Parameter Effectiveness** (`results/parameter_effectiveness.png`):
   - Heatmap showing which parameter combinations best distinguish between pain states
   - Bar chart showing which signal types are most sensitive to pain state changes
   - Helps identify optimal parameters for feature extraction

## Project Structure

```
AI4Pain/
├── data/              # Data directory
│   ├── train/         # Training data files
│   │   ├── Bvp/       # Blood Volume Pulse signals
│   │   ├── Eda/       # Electrodermal Activity signals
│   │   ├── Resp/      # Respiration signals
│   │   └── SpO2/      # Blood Oxygen Saturation signals
│   └── test/          # Test data files
│       ├── Bvp/       # Blood Volume Pulse signals
│       ├── Eda/       # Electrodermal Activity signals
│       ├── Resp/      # Respiration signals
│       └── SpO2/      # Blood Oxygen Saturation signals
├── logs/              # Log files directory
├── results/           # Output directory for feature tables and visualizations
├── src/               # Source code
│   └── feature_extraction.py  # Main feature extraction code
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── .gitignore         # Git ignore file
├── main.py            # Simple entry point script for running extraction
└── view_results.py    # Script for visualizing and analyzing feature data
```

## Data Organization

The project processes physiological signals organized by type:

- **Blood Volume Pulse (BVP)**: Measures changes in blood volume in peripheral tissues
- **Electrodermal Activity (EDA)**: Records skin conductance which varies with sweat gland activity
- **Respiration (RESP)**: Monitors breathing patterns and rate
- **Blood Oxygen Saturation (SpO₂)**: Measures oxygen levels in the blood

## Feature Extraction Process

The toolkit implements a multi-step feature extraction process:

1. **Signal Processing**: Reads physiological signals from CSV files organized by signal type
2. **Feature Calculation**:
   - **Permutation Entropy (PE)**: Quantifies the complexity/unpredictability of time series data
   - **Complexity Measure**: Derived from permutation entropy to assess system complexity
   - **Fisher Information**: Measures the amount of information a signal carries about pain states
3. **Parameter Exploration**: Analyzes each signal with multiple embedding dimensions (1-2 by default) and time delays (1-3 by default)
4. **State Identification**: Extracts pain state information (baseline/no pain, low, high) from signal metadata

All processing details are recorded in log files for traceability and debugging.

## Generated Features

The extraction generates comprehensive feature tables with the following information:

| Column | Description |
|--------|-------------|
| `file_name` | Source CSV file name |
| `signal` | Signal column identifier |
| `signal_type` | Type of signal (Bvp, Eda, Resp, SpO2) |
| `signallength` | Number of data points in the signal |
| `pe` | Permutation entropy value |
| `comp` | Complexity value |
| `fisher` | Fisher complexity value |
| `dimension` | Embedding dimension used |
| `tau` | Time delay used |
| `state` | Pain state (baseline/no pain, low, high) |

Optionally, Fisher PE columns can be included to validate Fisher PE ≈ Complexity PE.

## Technical Implementation

The tool is implemented in Python using:

- **NumPy**: For efficient numerical operations on signal data
- **Pandas**: For data manipulation and CSV handling
- **OrdPy**: For permutation entropy and complexity calculations
- **Matplotlib/Seaborn**: For visualization and analysis

The processing pipeline is designed to handle large datasets efficiently with comprehensive error handling and logging to ensure data quality.
