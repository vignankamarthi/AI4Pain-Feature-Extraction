#!/usr/bin/env python3
"""
Concatenated Silhouette Analysis for Pain Classification.

Instead of treating each time series column as an independent data point,
this script concatenates all columns of the same condition per subject into
one long time series, computes entropy features on the concatenated series,
and produces binary silhouette coefficients.

Per subject per signal (EDA, BVP):
  - All baseline columns -> 1 concatenated series (no-pain, label=0)
  - All low-pain columns -> 1 concatenated series (pain, label=1)
  - All high-pain columns -> 1 concatenated series (pain, label=1)

REST columns are excluded.

Outputs:
  - paper2_concatenated_entropy_pairs.csv  (entropy features per subject/signal/condition/d/tau)
  - paper2_concatenated_silhouette.csv     (silhouette scores per signal_type/d/tau/entropy_pair)

Usage:
    python extract_concatenated_silhouette.py
    python extract_concatenated_silhouette.py --data-dir data --output-dir results
    python extract_concatenated_silhouette.py --dimensions 3,4,5,6,7 --taus 1,2,3
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

ENTROPY_PAIRS = [
    ("CH", "pe", "comp"),
    ("Fisher", "fisher_shannon", "fisher_info"),
    ("Tsallis", "tsallispe", "tsalliscomp"),
    ("Renyi", "renyipe", "renyicomp"),
]

SIGNAL_TYPES = ["Eda", "Bvp"]
DATASETS = ["train", "validation"]

# Conditions to concatenate (REST excluded)
CONDITIONS = ["baseline", "low", "high"]

# Binary class mapping: baseline=0 (no-pain), low=1 (pain), high=1 (pain)
CONDITION_TO_BINARY = {
    "baseline": 0,
    "low": 1,
    "high": 1,
}


# ---------------------------------------------------------------------------
# Column grouping and concatenation
# ---------------------------------------------------------------------------

def group_columns_by_condition(df: pd.DataFrame, label_extractor: LabelExtractor) -> dict:
    """
    Group DataFrame columns by their pain condition.

    Args:
        df: Subject signal DataFrame (rows=time samples, columns=signal channels)
        label_extractor: LabelExtractor instance for parsing column names

    Returns:
        Dict mapping condition -> list of column names.
        Only includes baseline, low, high (REST and unknown excluded).
    """
    groups = {"baseline": [], "low": [], "high": []}

    for col_name in df.columns:
        state = label_extractor.extract_state(str(col_name))
        if state in groups:
            groups[state].append(col_name)

    return groups


def concatenate_condition_signals(df: pd.DataFrame, columns: list) -> np.ndarray:
    """
    Concatenate multiple signal columns into one long time series.

    For each column, strips NaN values first, then concatenates.

    Args:
        df: Subject signal DataFrame
        columns: List of column names to concatenate

    Returns:
        1D numpy array of concatenated non-NaN values.
        Empty array if no valid data.
    """
    segments = []
    for col in columns:
        data = df[col].values
        clean = data[~np.isnan(data)]
        if len(clean) > 0:
            segments.append(clean)

    if not segments:
        return np.array([])

    return np.concatenate(segments)


# ---------------------------------------------------------------------------
# Silhouette computation (reused from extract_paper2.py)
# ---------------------------------------------------------------------------

def compute_silhouette_indices(entropy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute binary silhouette scores for each (signal_type, d, tau, entropy_pair).

    Uses binaryclass column directly (already binary: 0=no-pain, 1=pain).

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
# Data loading helpers (reused from extract_paper2.py)
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_concatenated_extraction(
    data_dir: str = "data",
    output_dir: str = "results",
    dimensions: list = None,
    taus: list = None,
    apply_z_score: bool = True,
):
    """
    Run concatenated silhouette analysis pipeline.

    For each subject and signal type:
      1. Group columns by condition (baseline, low, high)
      2. Concatenate all columns within each condition into one long series
      3. Z-score normalize each concatenated series
      4. Compute entropy features at each (d, tau)
      5. Compute binary silhouette (baseline=0 vs low+high=1)

    Outputs:
      - paper2_concatenated_entropy_pairs.csv
      - paper2_concatenated_silhouette.csv
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

    for dataset in DATASETS:
        eda_dir = data_path / dataset / "Eda"
        bvp_dir = data_path / dataset / "Bvp"

        if not eda_dir.exists() or not bvp_dir.exists():
            print(f"Skipping {dataset}: missing Eda or Bvp directory")
            continue

        eda_files = sorted(
            eda_dir.glob("*.csv"),
            key=lambda p: _extract_subject_number(p.name),
        )

        for eda_file in eda_files:
            subject_id = _extract_subject_number(eda_file.name)
            bvp_file = bvp_dir / eda_file.name

            print(f"Processing subject {subject_id} ({dataset})...")

            eda_df = _load_csv_safe(eda_file)
            bvp_df = _load_csv_safe(bvp_file)

            if eda_df.empty and bvp_df.empty:
                print(f"  Skipping subject {subject_id}: both files empty")
                continue

            for sig_type, sig_df in [("Eda", eda_df), ("Bvp", bvp_df)]:
                if sig_df.empty:
                    continue

                # Group columns by condition
                col_groups = group_columns_by_condition(sig_df, label_extractor)

                for condition in CONDITIONS:
                    columns = col_groups[condition]
                    if not columns:
                        print(f"  No {condition} columns for subject {subject_id} {sig_type}")
                        continue

                    # Concatenate all columns of this condition
                    concat_signal = concatenate_condition_signals(sig_df, columns)

                    if len(concat_signal) == 0:
                        print(f"  Empty concatenated signal for {subject_id} {sig_type} {condition}")
                        continue

                    # Z-score normalization on concatenated series
                    if apply_z_score and np.std(concat_signal) > 0:
                        concat_signal = (
                            (concat_signal - np.mean(concat_signal))
                            / np.std(concat_signal)
                        )

                    binary_label = CONDITION_TO_BINARY[condition]

                    # Compute entropy at each (d, tau)
                    for d in dimensions:
                        for tau in taus:
                            entropy_dict = calculator.calculate_all_entropies(
                                concat_signal, d, tau
                            )

                            all_entropy_rows.append({
                                "subject": subject_id,
                                "dataset": dataset,
                                "signal_type": sig_type,
                                "condition": condition,
                                "binaryclass": binary_label,
                                "n_columns_concatenated": len(columns),
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
                                "signallength": len(concat_signal),
                            })

                    gc.collect()

    if not all_entropy_rows:
        print("No entropy data produced. Check data directory.")
        return

    entropy_df = pd.DataFrame(all_entropy_rows)

    # Save entropy pairs CSV
    entropy_out = output_path / "paper2_concatenated_entropy_pairs.csv"
    entropy_df.to_csv(entropy_out, index=False)
    print(f"\nSaved concatenated entropy pairs: {entropy_out} ({len(entropy_df)} rows)")

    # Compute silhouette indices
    silhouette_df = compute_silhouette_indices(entropy_df)

    silhouette_out = output_path / "paper2_concatenated_silhouette.csv"
    silhouette_df.to_csv(silhouette_out, index=False)
    print(f"Saved concatenated silhouette: {silhouette_out} ({len(silhouette_df)} rows)")

    # Print summary
    if not silhouette_df.empty:
        print("\nTop concatenated silhouette scores:")
        valid = silhouette_df.dropna(subset=["silhouette_score"])
        if not valid.empty:
            top = valid.nlargest(10, "silhouette_score")
            for _, row in top.iterrows():
                print(
                    f"  {row['signal_type']} d={row['dimension']} tau={row['tau']} "
                    f"{row['entropy_pair']}: {row['silhouette_score']:.4f} "
                    f"(n={row['n_samples']})"
                )

    # Print condition counts per signal type
    print("\nData point counts per signal type:")
    for sig in SIGNAL_TYPES:
        sig_data = entropy_df[
            (entropy_df["signal_type"] == sig)
            & (entropy_df["dimension"] == dimensions[0])
            & (entropy_df["tau"] == taus[0])
        ]
        n_nopain = (sig_data["binaryclass"] == 0).sum()
        n_pain = (sig_data["binaryclass"] == 1).sum()
        print(f"  {sig}: {n_nopain} no-pain + {n_pain} pain = {n_nopain + n_pain} total")


def main():
    parser = argparse.ArgumentParser(
        description="Concatenated Silhouette Analysis for Pain Classification"
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

    run_concatenated_extraction(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dimensions=dimensions,
        taus=taus_list,
        apply_z_score=not args.no_zscore,
    )


if __name__ == "__main__":
    main()
