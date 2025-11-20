# AI4Pain Feature Extraction V2

**Entropy-Based Feature Extraction for Physiological Pain Assessment**

Production-ready Python implementation extracting entropy-based features from physiological signals (BVP, EDA, Resp, SpO2) for multi-class pain state classification. Long-format output aligned with Jupyter notebook analysis pipeline.

---

## Overview

This system implements five entropy-based complexity measures across multiple temporal scales to characterize physiological responses to pain:

- **8 entropy measures**: Permutation Entropy, Statistical Complexity, Fisher-Shannon, Fisher Information, Renyi Entropy/Complexity, Tsallis Entropy/Complexity
- **Long-format output**: 15 rows per signal (5 dimensions × 3 time delays), 16 columns per row
- **Multi-signal processing**: BVP, EDA, Respiration, SpO2
- **4-class pain states**: Baseline, Low, High, Rest (+ unknown handling)
- **Robust preprocessing**: Z-score normalization, NaN tracking, granular file organization

---

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd AI4Pain-Feature-Extraction-V2

# Create virtual environment (Python 3.8+)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Organization

Structure your data as follows:
```
data/
├── train/
│   ├── Bvp/*.csv
│   ├── Eda/*.csv
│   ├── Resp/*.csv
│   └── SpO2/*.csv
├── validation/
│   └── [same structure]
└── test/
    └── [same structure]
```

**CSV format**: Each column = one participant trial, rows = time samples

**Note**: The `data/test/`, `data/train/`, and `data/validation/` directories are preserved as placeholders in the repository via `.gitkeep` files. All data files within these directories are gitignored for data integrity. 

### Auto-Generated Folders

The following folders are created automatically during execution and are gitignored:

**`logs/`** - Contains execution logs (gitignored)

**`results/`** - Contains output CSV files and visualizations (gitignored)

### Basic Usage

```bash
# Process all datasets, all signal types (default)
python main.py

# Process specific dataset(s) - space-separated
python main.py --dataset train
python main.py --dataset train validation

# Process specific signal type(s) - space-separated
python main.py --signal-type Bvp Eda
python main.py --dataset train --signal-type Bvp

# Custom dimensions and time delays
python main.py --dimensions 4,5,6 --taus 1,2

# Disable z-score normalization
python main.py --no-zscore

# Disable progress bars
python main.py --no-progress
```

### Viewing Results

```bash
# Generate notebook-style visualization (5×3 Renyi complexity-entropy grid)
python visualize_notebook_style.py --dataset train

# Specify signal types for visualization
python visualize_notebook_style.py --dataset validation --signal-type Bvp Eda
```

---

## Architecture

```
src/
├── config/
│   └── settings.py              # Configuration management
├── core/
│   ├── entropy_calculator.py    # 8 entropy implementations (ordpy-based)
│   └── feature_extractor.py     # Long-format orchestration
├── preprocessing/
│   ├── label_extractor.py       # State and binaryclass extraction
│   └── signal_processor.py      # Z-score normalization
└── utils/
    ├── data_loader.py            # CSV I/O
    └── logger.py                 # Structured logging
```

**Entry points:**
- `main.py`: CLI interface with multi-select support
- `visualize_notebook_style.py`: 5×3 Renyi complexity-entropy visualization

---

## Entropy Measures

### 1. Permutation Entropy (PE)
Quantifies complexity of ordinal patterns in time series. Lower values indicate more predictable signals.

**Formula**: Shannon entropy of ordinal pattern distribution, normalized by log(d!)

### 2. Statistical Complexity (C)
Measures departure from equilibrium distribution. Captures structural complexity beyond simple randomness.

**Formula**: C = PE × JS-divergence (pattern dist || uniform dist)

### 3. Fisher Information (fisher_info, fisher_shannon)
Measures local sensitivity to parameter changes.
- **fisher_shannon**: Shannon-weighted Fisher information (ordpy index [0])
- **fisher_info**: Gradient-based sensitivity (ordpy index [1])

### 4. Renyi Entropy (Renyi_PE, Renyi_C)
Generalized entropy with parameter q=1 (Shannon limit).

**Formula**: H_q = (1/(1-q)) × log(Σ p_i^q)

### 5. Tsallis Entropy (Tsallis_PE, Tsallis_C)
Non-extensive entropy for long-range correlations (q=1, Shannon limit).

**Formula**: S_q = (1/(q-1)) × (1 - Σ p_i^q)

---

## Output

**Format**: Long-format CSV (15 rows per signal, 16 columns per row)

**File Pattern**: `results/results_{dataset}_{signal_type}.csv`
- Example: `results_train_bvp.csv`, `results_validation_eda.csv`

**Columns (16 total)**:
1. `file_name`: Full path (`data/train/Bvp/1.csv`)
2. `signal`: Column name from source CSV (`1_Baseline_1`)
3. `signallength`: Valid samples after NaN removal
4. `pe`: Permutation Entropy
5. `comp`: Statistical Complexity
6. `fisher_shannon`: Fisher-Shannon entropy
7. `fisher_info`: Fisher Information
8. `renyipe`: Renyi Permutation Entropy
9. `renyicomp`: Renyi Complexity
10. `tsallispe`: Tsallis Permutation Entropy
11. `tsalliscomp`: Tsallis Complexity
12. `dimension`: Embedding dimension (3-7)
13. `tau`: Time delay (1-3)
14. `state`: Pain state (baseline/low/high/rest/unknown)
15. `binaryclass`: Numeric encoding (0/1/2/3/-1)
16. `nan_percentage`: NaN percentage in original signal

**Example rows**:
```csv
file_name,signal,signallength,pe,comp,fisher_shannon,fisher_info,renyipe,renyicomp,tsallispe,tsalliscomp,dimension,tau,state,binaryclass,nan_percentage
data/train/Bvp/6.csv,6_Baseline_1,5950,0.999332,0.000651,0.999332,0.000437,0.999332,0.000651,0.999332,0.000651,3,1,baseline,0,0.00
data/train/Bvp/6.csv,6_Baseline_1,5950,0.997424,0.002532,0.997424,0.001581,0.997424,0.002532,0.997424,0.002532,3,2,baseline,0,0.00
data/train/Bvp/6.csv,6_Baseline_1,5950,0.992063,0.007686,0.992063,0.004582,0.992063,0.007686,0.992063,0.007686,3,3,baseline,0,0.00
...
```

**Structure**: Each signal generates 15 rows (5 dimensions × 3 taus)

---

## Performance

**Hardware**: MacBook Pro M1, 8 cores, 16 GB RAM

**Metrics**:
- **Processing speed**: ~1.82 samples/second
- **Total runtime**: ~4332 seconds for 7,872 samples
- **Memory usage**: ~2-3 GB peak

**For 201× speedup with identical results, see [Rust implementation](https://github.com/vignankamarthi/Feature-Extraction-Rust)**

---

## Dependencies

**Core libraries**:
- `ordpy==1.1.0`: Ordinal pattern analysis (reference implementation)
- `numpy>=1.24.0`: Numerical operations
- `pandas>=2.0.0`: DataFrame I/O
- `scipy>=1.10.0`: Signal filtering (optional)

**CLI and logging**:
- `tqdm>=4.65.0`: Progress bars

**Full list**: See `requirements.txt`

---

## Configuration

**Command-line arguments** (see `python main.py --help`):
- `--dataset`: Space-separated datasets (train validation test), default: all three
- `--signal-type`: Space-separated signal types (Bvp Eda Resp SpO2), default: all four
- `--dimensions`: Comma-separated embedding dimensions (default: 3,4,5,6,7)
- `--taus`: Comma-separated time delays (default: 1,2,3)
- `--no-zscore`: Disable z-score normalization (default: on)
- `--no-progress`: Disable progress bars (default: on)

**Programmatic configuration**: Use `src/config/settings.py` Settings class

---

## Project Structure

```
AI4Pain-Feature-Extraction-V2/
├── data/                         # Input directory (gitignored - private)
│   ├── test/.gitkeep             # Placeholder for test data
│   ├── train/.gitkeep            # Placeholder for train data
│   └── validation/.gitkeep       # Placeholder for validation data
├── logs/                         # Auto-generated logs (gitignored)
├── results/                      # Output CSVs and plots (gitignored)
├── src/                          # Source code (see Architecture above)
├── main.py                       # CLI entry point with multi-select
├── visualize_notebook_style.py  # 5×3 Renyi complexity-entropy plots
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes data/, logs/, results/
└── README.md                     # This file
```
---

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'ordpy'`
- **Solution**: `pip install -r requirements.txt`

**Issue**: `ValueError: Signal too short after NaN removal`
- **Solution**: Signals with >85% NaN are skipped automatically. Check data quality.

**Issue**: Slow processing (>1 hour for training set)
- **Expected**: Python processes ~1.82 samples/sec. For 145× speedup, use [Rust implementation](../ai4pain-rust/).

**Issue**: `FileNotFoundError: data/ directory not found`
- **Solution**: Create data directory with required structure (see Quick Start above)

---

## Changes from Previous Versions

**V2 Long-Format Update** (Current):
- 16-column long-format output (15 rows per signal)
- Multi-select CLI (space-separated datasets and signal types)
- Granular file organization (`results_{dataset}_{signal_type}.csv`)
- Notebook-aligned visualization (5×3 Renyi complexity-entropy grid)
- 4-class pain states with explicit state labels + binary encoding

**Previous Wide-Format**:
- 120+-column wide format (1 row per signal)
- Single-select CLI
- Concatenated output files

---

## References

1. **Permutation Entropy**: Bandt, C., & Pompe, B. (2002). *Physical Review Letters*, 88(17), 174102.
2. **Statistical Complexity**: Rosso, O. A., et al. (2007). *Physical Review Letters*, 99(15), 154102.
3. **Fisher Information**: Martin, M. T., Plastino, A., & Rosso, O. A. (2003). *Physics Letters A*, 311(2-3), 126-132.
4. **Ordpy Library**: Pessa, A. A. B., & Ribeiro, H. V. (2021). *Chaos*, 31(6), 063110.

---

## Citation

```bibtex
@software{ai4pain_v2,
  author = {Kamarthi, Vignan},
  title = {AI4Pain Feature Extraction V2: Entropy-Based Physiological Signal Analysis},
  year = {2025},
  institution = {Northeastern University}
}
```