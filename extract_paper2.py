#!/usr/bin/env python3
"""
Paper 2 Feature Extraction: Ordinal Pattern Analysis for Pain Classification.

Extracts entropy-complexity pairs from EDA + BVP signals with silhouette analysis.
Produces two outputs:
  - paper2_entropy_pairs.csv  (all entropy measures per signal per d,tau)
  - paper2_silhouette_index.csv  (silhouette scores per signal_type, d, tau, pair)

Usage:
    python extract_paper2.py
    python extract_paper2.py --include-rest
    python extract_paper2.py --dimensions 3,4,5,6,7 --taus 1,2,3
    python extract_paper2.py --data-dir data --output-dir results
"""

import argparse
import sys
import gc
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent))

from src.core.entropy_calculator import EntropyCalculator
from src.preprocessing.label_extractor import LabelExtractor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Each tuple: (pair_name, x_column, y_column)
ENTROPY_PAIRS = [
    ("CH", "pe", "comp"),
    ("Fisher", "fisher_shannon", "fisher_info"),
    ("Tsallis", "tsallispe", "tsalliscomp"),
    ("Renyi", "renyipe", "renyicomp"),
]

SIGNAL_TYPES = ["Eda", "Bvp"]
DATASETS = ["train", "validation"]


# ---------------------------------------------------------------------------
# Per-subject minimum length truncation
# ---------------------------------------------------------------------------

def find_subject_min_length(eda_df: pd.DataFrame, bvp_df: pd.DataFrame) -> int:
    """
    Find minimum non-NaN column length across both EDA and BVP for a subject.

    For each column, count non-NaN values. Return the minimum across ALL columns
    in BOTH DataFrames. This ensures all signals for a subject have equal length.

    Returns 0 if either DataFrame is empty.
    """
    if eda_df.empty or bvp_df.empty:
        return 0

    lengths = []
    for df in [eda_df, bvp_df]:
        for col in df.columns:
            lengths.append(df[col].notna().sum())

    return int(min(lengths)) if lengths else 0


def truncate_to_length(df: pd.DataFrame, length: int) -> pd.DataFrame:
    """Truncate all columns to specified length (drop trailing rows)."""
    if df.empty or length <= 0:
        return df.iloc[0:0]  # preserve columns, zero rows
    return df.iloc[:length].copy()


# ---------------------------------------------------------------------------
# Class filtering
# ---------------------------------------------------------------------------

def filter_classes(df: pd.DataFrame, include_rest: bool = False) -> pd.DataFrame:
    """
    Filter DataFrame rows by pain class.

    With include_rest=False:
        Keep baseline(0), low(1), high(2). Drop rest(3).
    With include_rest=True:
        Keep all. Map rest(3) -> 0 (no-pain alongside baseline).

    Returns a copy with updated binaryclass column.
    """
    result = df.copy()

    if include_rest:
        result.loc[result["binaryclass"] == 3, "binaryclass"] = 0
    else:
        result = result[result["binaryclass"] != 3].copy()

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Silhouette computation
# ---------------------------------------------------------------------------

def compute_silhouette_indices(entropy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute silhouette scores for each (signal_type, d, tau, entropy_pair).

    For each pair, use the 2D feature space (x_col, y_col) and class labels.
    Total expected: n_signals x n_d x n_tau x 4 pairs.

    Returns DataFrame with columns:
        signal_type, dimension, tau, entropy_pair, feature_x, feature_y,
        silhouette_score, n_samples, n_classes
    """
    results = []

    groups = entropy_df.groupby(["signal_type", "dimension", "tau"])

    for (sig_type, dim, tau), group in groups:
        labels = group["binaryclass"].values
        n_classes = len(np.unique(labels))
        n_samples = len(labels)

        for pair_name, x_col, y_col in ENTROPY_PAIRS:
            features = group[[x_col, y_col]].values

            # Silhouette requires >= 2 classes and >= 2 samples
            valid_mask = ~np.isnan(features).any(axis=1)
            features_clean = features[valid_mask]
            labels_clean = labels[valid_mask]
            n_classes_clean = len(np.unique(labels_clean))
            n_samples_clean = len(labels_clean)

            if n_classes_clean < 2 or n_samples_clean < 2:
                score = np.nan
            else:
                try:
                    score = silhouette_score(features_clean, labels_clean)
                except Exception:
                    score = np.nan

            results.append({
                "signal_type": sig_type,
                "dimension": dim,
                "tau": tau,
                "entropy_pair": pair_name,
                "feature_x": x_col,
                "feature_y": y_col,
                "silhouette_score": score,
                "n_samples": n_samples_clean,
                "n_classes": n_classes_clean,
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _extract_subject_number(filename: str) -> int:
    """Extract subject number from filename like '1.csv' -> 1."""
    numbers = re.findall(r"\d+", filename)
    return int(numbers[0]) if numbers else -1


def _load_csv_safe(path: Path) -> pd.DataFrame:
    """Load CSV with error handling. Returns empty DataFrame on failure."""
    try:
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def _get_subject_ids(data_dir: Path) -> set:
    """Collect all subject IDs across EDA and BVP in train + validation."""
    ids = set()
    for dataset in DATASETS:
        for sig in SIGNAL_TYPES:
            sig_dir = data_dir / dataset / sig
            if sig_dir.exists():
                for f in sig_dir.glob("*.csv"):
                    sid = _extract_subject_number(f.name)
                    if sid >= 0:
                        ids.add((dataset, sid, f.stem))
    return ids


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_extraction(
    data_dir: str = "data",
    output_dir: str = "results",
    dimensions: list = None,
    taus: list = None,
    include_rest: bool = False,
    apply_z_score: bool = True,
):
    """
    Run the full Paper 2 extraction pipeline.

    Steps:
      1. Load EDA + BVP from train + validation
      2. Per-subject truncation (min non-NaN length across both signals)
      3. Compute all 8 entropy measures for each signal at each (d, tau)
      4. Filter classes (with or without REST)
      5. Compute silhouette indices for each (signal_type, d, tau, entropy_pair)
      6. Save outputs
    """
    if dimensions is None:
        dimensions = [3, 4, 5, 6, 7]
    if taus is None:
        taus = [1, 2, 3]

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    calculator = EntropyCalculator()
    label_extractor = LabelExtractor()

    all_entropy_rows = []

    # Process each dataset split
    for dataset in DATASETS:
        # Find all subject CSV files in EDA directory
        eda_dir = data_path / dataset / "Eda"
        bvp_dir = data_path / dataset / "Bvp"

        if not eda_dir.exists() or not bvp_dir.exists():
            print(f"Skipping {dataset}: missing Eda or Bvp directory")
            continue

        eda_files = sorted(eda_dir.glob("*.csv"), key=lambda p: _extract_subject_number(p.name))

        for eda_file in eda_files:
            subject_id = _extract_subject_number(eda_file.name)
            bvp_file = bvp_dir / eda_file.name

            print(f"Processing subject {subject_id} ({dataset})...")

            # Load both signal types
            eda_df = _load_csv_safe(eda_file)
            bvp_df = _load_csv_safe(bvp_file)

            if eda_df.empty and bvp_df.empty:
                print(f"  Skipping subject {subject_id}: both files empty")
                continue

            # Per-subject minimum length truncation
            min_len = find_subject_min_length(eda_df, bvp_df)
            if min_len == 0:
                print(f"  Skipping subject {subject_id}: min length is 0")
                continue

            eda_trunc = truncate_to_length(eda_df, min_len)
            bvp_trunc = truncate_to_length(bvp_df, min_len)

            # Process each signal type
            for sig_type, sig_df in [("Eda", eda_trunc), ("Bvp", bvp_trunc)]:
                for col_name in sig_df.columns:
                    signal_data = sig_df[col_name].values
                    clean_signal = signal_data[~np.isnan(signal_data)]

                    if len(clean_signal) == 0:
                        continue

                    # Z-score normalization
                    if apply_z_score and np.std(clean_signal) > 0:
                        clean_signal = (clean_signal - np.mean(clean_signal)) / np.std(clean_signal)

                    # Extract state and class label
                    state, binaryclass = label_extractor.extract_labels(col_name)

                    # Compute all entropy measures across (d, tau) grid
                    for d in dimensions:
                        for tau in taus:
                            entropy_dict = calculator.calculate_all_entropies(
                                clean_signal, d, tau
                            )

                            all_entropy_rows.append({
                                "subject": subject_id,
                                "signal_type": sig_type,
                                "signal_name": col_name,
                                "state": state,
                                "binaryclass": binaryclass,
                                "dimension": d,
                                "tau": tau,
                                "pe": entropy_dict["pe"],
                                "comp": entropy_dict["comp"],
                                "fisher_shannon": entropy_dict["fisher_shannon"],
                                "fisher_info": entropy_dict["fisher_info"],
                                "renyipe": entropy_dict["renyipe"],
                                "renyicomp": entropy_dict["renyicomp"],
                                "tsallispe": entropy_dict["tsallispe"],
                                "tsalliscomp": entropy_dict["tsalliscomp"],
                                "signallength": len(clean_signal),
                            })

                    gc.collect()

    if not all_entropy_rows:
        print("No entropy data produced. Check data directory.")
        return

    entropy_df = pd.DataFrame(all_entropy_rows)

    # Filter classes
    entropy_filtered = filter_classes(entropy_df, include_rest=include_rest)

    # Save entropy pairs CSV
    entropy_out = output_path / "paper2_entropy_pairs.csv"
    entropy_filtered.to_csv(entropy_out, index=False)
    print(f"\nSaved entropy pairs: {entropy_out} ({len(entropy_filtered)} rows)")

    # Compute silhouette indices
    silhouette_df = compute_silhouette_indices(entropy_filtered)

    silhouette_out = output_path / "paper2_silhouette_index.csv"
    silhouette_df.to_csv(silhouette_out, index=False)
    print(f"Saved silhouette index: {silhouette_out} ({len(silhouette_df)} rows)")

    # Print best silhouette scores
    if not silhouette_df.empty:
        print("\nTop silhouette scores:")
        valid = silhouette_df.dropna(subset=["silhouette_score"])
        if not valid.empty:
            top = valid.nlargest(10, "silhouette_score")
            for _, row in top.iterrows():
                print(
                    f"  {row['signal_type']} d={row['dimension']} tau={row['tau']} "
                    f"{row['entropy_pair']}: {row['silhouette_score']:.4f}"
                )


def main():
    parser = argparse.ArgumentParser(description="Paper 2 Feature Extraction")
    parser.add_argument(
        "--include-rest",
        action="store_true",
        help="Include REST periods as no-pain class",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        default="3,4,5,6,7",
        help="Comma-separated embedding dimensions (default: 3,4,5,6,7)",
    )
    parser.add_argument(
        "--taus",
        type=str,
        default="1,2,3",
        help="Comma-separated time delays (default: 1,2,3)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Path to output directory (default: results)",
    )
    parser.add_argument(
        "--no-zscore",
        action="store_true",
        help="Disable z-score normalization",
    )
    args = parser.parse_args()

    dimensions = [int(d.strip()) for d in args.dimensions.split(",")]
    taus_list = [int(t.strip()) for t in args.taus.split(",")]

    run_extraction(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dimensions=dimensions,
        taus=taus_list,
        include_rest=args.include_rest,
        apply_z_score=not args.no_zscore,
    )


if __name__ == "__main__":
    main()
