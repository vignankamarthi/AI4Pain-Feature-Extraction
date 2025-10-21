#!/usr/bin/env python3
"""
Clean Python extraction run for validation.
Saves output with 'python_' prefix to distinguish from Rust.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import ordpy
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
# Disable SystemLogger to prevent performance degradation
# from src.preprocessing.signal_processor import SignalPreprocessor
# from src.utils.logger import SystemLogger

def extract_entropy_features(signal: np.ndarray, dimension: int, tau: int) -> Dict[str, float]:
    """
    Extract all five entropy measures using ordpy library.

    Calculates Permutation Entropy (PE), Statistical Complexity (C),
    Fisher Information, Renyi Entropy, and Tsallis Entropy for a signal.

    Parameters
    ----------
    signal : np.ndarray
        Input time series data (1D array).
    dimension : int
        Embedding dimension (dx parameter in ordpy).
    tau : int
        Time delay (taux parameter in ordpy).

    Returns
    -------
    Dict[str, float]
        Dictionary containing entropy features with keys:
        - PE_d{dimension}_t{tau} : float
            Permutation entropy value
        - C_d{dimension}_t{tau} : float
            Statistical complexity value
        - Fisher_S_d{dimension}_t{tau} : float
            Fisher-Shannon entropy
        - Fisher_I_d{dimension}_t{tau} : float
            Fisher information measure
        - Renyi_C_d{dimension}_t{tau} : float
            Renyi complexity
        - Renyi_PE_d{dimension}_t{tau} : float
            Renyi permutation entropy
        - Tsallis_C_d{dimension}_t{tau} : float
            Tsallis complexity
        - Tsallis_PE_d{dimension}_t{tau} : float
            Tsallis permutation entropy

    Notes
    -----
    - Removes NaN values before calculation
    - Returns empty dict if signal length < 100
    - All exceptions are caught and logged silently
    """
    features = {}
    try:
        clean_signal = signal[~np.isnan(signal)]
        if len(clean_signal) < 100:
            return {}

        # All 5 entropy types
        pe, c = ordpy.complexity_entropy(clean_signal, dx=dimension, taux=tau)
        features[f'PE_d{dimension}_t{tau}'] = pe
        features[f'C_d{dimension}_t{tau}'] = c

        fisher_s, fisher_i = ordpy.fisher_shannon(clean_signal, dx=dimension, taux=tau)
        features[f'Fisher_S_d{dimension}_t{tau}'] = fisher_s
        features[f'Fisher_I_d{dimension}_t{tau}'] = fisher_i

        renyi_c, renyi_pe = ordpy.renyi_complexity_entropy(clean_signal, dx=dimension, taux=tau)
        features[f'Renyi_C_d{dimension}_t{tau}'] = renyi_c
        features[f'Renyi_PE_d{dimension}_t{tau}'] = renyi_pe

        tsallis_c, tsallis_pe = ordpy.tsallis_complexity_entropy(clean_signal, dx=dimension, taux=tau)
        features[f'Tsallis_C_d{dimension}_t{tau}'] = tsallis_c
        features[f'Tsallis_PE_d{dimension}_t{tau}'] = tsallis_pe

    except Exception as e:
        pass

    return features

def z_score_normalize(signal: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization to standardize signal.

    Transforms signal to have zero mean and unit variance.

    Parameters
    ----------
    signal : np.ndarray
        Input signal array.

    Returns
    -------
    np.ndarray
        Z-score normalized signal. If std is zero, returns mean-centered signal.

    Notes
    -----
    Normalization formula: (signal - mean) / std
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return signal - mean
    return (signal - mean) / std

def process_column(signal: np.ndarray, column_name: str, signal_type: str, dataset: str) -> Optional[Dict]:
    """
    Process a single CSV column to extract entropy features.

    Performs z-score normalization and calculates entropy features across
    all dimension-tau combinations (d=3-7, tau=1-3).

    Parameters
    ----------
    signal : np.ndarray
        Raw signal data from CSV column.
    column_name : str
        Name of the CSV column (used for file_id and label extraction).
    signal_type : str
        Type of physiological signal ('Bvp', 'Eda', 'Resp', 'SpO2').
    dataset : str
        Dataset name ('train', 'validation', or 'test').

    Returns
    -------
    Optional[Dict]
        Dictionary containing all features and metadata, or None if signal
        is too short (<100 samples after NaN removal).

        Output dictionary keys:
        - All entropy features (PE_*, C_*, Fisher_*, Renyi_*, Tsallis_*)
        - file_id : str
        - label : int (0=baseline, 1=pain)
        - dataset : str
        - signal_type : str
        - signal_length : int
        - nan_percentage : float

    Notes
    -----
    Label extraction logic:
    - 'baseline' or 'rest' in name → label=0
    - 'low' or 'high' in name → label=1
    - Otherwise → label=-1
    """
    # Calculate NaN percentage
    nan_pct = (np.isnan(signal).sum() / len(signal)) * 100

    # Clean and normalize
    clean_signal = signal[~np.isnan(signal)]
    if len(clean_signal) < 100:
        return None

    normalized = z_score_normalize(clean_signal)

    # Extract all entropy features
    all_features = {}
    for dim in [3, 4, 5, 6, 7]:
        for tau in [1, 2, 3]:
            features = extract_entropy_features(normalized, dim, tau)
            all_features.update(features)

    if not all_features:
        return None

    # Extract label
    name_lower = column_name.lower()
    if 'baseline' in name_lower or 'rest' in name_lower:
        label = 0
    elif 'low' in name_lower or 'high' in name_lower:
        label = 1
    else:
        label = -1

    # Add metadata
    all_features['file_id'] = column_name
    all_features['label'] = label
    all_features['dataset'] = dataset
    all_features['signal_type'] = signal_type
    all_features['signal_length'] = len(clean_signal)
    all_features['nan_percentage'] = nan_pct

    return all_features

def extract_dataset(dataset: str = 'train'):
    """
    Extract entropy features from all signals in a dataset.

    Processes all signal types (Bvp, Eda, Resp, SpO2) in the specified
    dataset directory and saves results to CSV.

    Parameters
    ----------
    dataset : str, default='train'
        Dataset to process ('train', 'validation', or 'test').

    Returns
    -------
    pd.DataFrame or None
        DataFrame containing all extracted features and metadata, or None
        if no valid samples were extracted.

    Notes
    -----
    - Reads CSV files from: data/{dataset}/{signal_type}/*.csv
    - Saves output to: results/python_features_{dataset}.csv
    - Prints progress and performance statistics to console
    - Processes dimensions [3,4,5,6,7] and taus [1,2,3] for each sample
    """
    print(f"\n{'='*60}")
    print(f"PYTHON EXTRACTION: {dataset.upper()}")
    print(f"{'='*60}\n")

    data_dir = Path('data') / dataset
    signal_types = ['Bvp', 'Eda', 'Resp', 'SpO2']

    all_results = []
    start_time = time.time()

    for signal_type in signal_types:
        signal_dir = data_dir / signal_type
        if not signal_dir.exists():
            continue

        files = list(signal_dir.glob('*.csv'))
        print(f"Processing {signal_type}: {len(files)} files")

        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                for column in df.columns:
                    signal = df[column].values
                    result = process_column(signal, column, signal_type, dataset)
                    if result:
                        all_results.append(result)
            except Exception as e:
                print(f"  Error: {file_path.name}: {e}")

        print(f"  Extracted {len([r for r in all_results if r['signal_type'] == signal_type])} {signal_type} samples")

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        output_file = Path('results') / f'python_features_{dataset}.csv'
        os.makedirs('results', exist_ok=True)
        df.to_csv(output_file, index=False)

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ PYTHON EXTRACTION COMPLETE")
        print(f"  Total samples: {len(df)}")
        print(f"  Time: {elapsed:.2f}s ({len(df)/elapsed:.1f} samples/s)")
        print(f"  Output: {output_file}")
        print(f"{'='*60}\n")

        return df

    return None

if __name__ == "__main__":
    extract_dataset('train')
