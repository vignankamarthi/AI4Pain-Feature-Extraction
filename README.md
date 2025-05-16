# AI4PAIN 2025 Feature Extraction

This project provides a feature extraction toolkit for the AI4PAIN 2025 Challenge, focusing on physiological signals (EDA, BVP, RESP, SpO₂) for pain assessment.

## Project Structure

```
AI4Pain/
├── data/              # Data directory
│   ├── train/         # Training data files
│   └── test/          # Test data files
├── logs/              # Log files directory
├── results/           # Output directory for feature tables
├── src/               # Source code
│   └── feature_extraction.py  # Main feature extraction code
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── run_extraction.py  # Convenience script to run extraction
```

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your CSV files in the appropriate directories:
   - Training data: `data/train/`
   - Test data: `data/test/`

## Running Feature Extraction

To run the feature extraction with default settings:

```bash
python run_extraction.py
```

To include PE verification column in the output (for debugging):

```bash
python run_extraction.py --verify
```

## Output

The script will generate three CSV files in the `results` directory:

1. `features_train.csv` - Features extracted from training data
2. `features_test.csv` - Features extracted from test data
3. `features_complete.csv` - Combined features from both datasets

The resulting table contains the following columns:

- `file_name`: Name of the source CSV file
- `signal`: Signal column name from the source file
- `signallength`: Number of data points in the signal
- `pe`: Permutation entropy value
- `comp`: Complexity value
- `fisher`: Fisher complexity value
- `dimension`: Embedding dimension used (4 or 5)
- `tau`: Time delay used (1, 2, or 3)
- `state`: Pain state extracted from signal name (baseline, low, high)

When run with the `--verify` flag, an additional column `pe_fisher` is included for verification.

## Logging

All processing details and errors are logged to `logs/feature_extraction.log`. The log file is reset at the start of each run.

## Implementation Details

The feature extraction process:

1. Reads all CSV files from the data directories
2. For each signal column in each file:
   - Extracts the signal data
   - Applies permutation entropy and complexity calculations with different dimension and tau values
   - Determines the pain state from the signal name
3. Compiles all results into feature tables
4. Saves the results as CSV files

## Requirements

- Python 3.7+
- numpy
- pandas
- ordpy
