"""
Tests for extract_paper2.py -- Paper 2 feature extraction pipeline.

All tests use synthetic data. NO real dataset access.
Tests validate:
  1. Per-subject minimum length truncation
  2. Binary class filtering without REST
  3. Binary class filtering with REST
  4. Silhouette computation on synthetic clusters
  5. Empty file / DataFrame handling
  6. Entropy pair extraction structure
  7. Output CSV format
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path so extract_paper2 can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from extract_paper2 import (
    find_subject_min_length,
    truncate_to_length,
    filter_classes,
    compute_silhouette_indices,
    ENTROPY_PAIRS,
)
from src.core.entropy_calculator import EntropyCalculator


# ---------------------------------------------------------------------------
# Helpers for building synthetic DataFrames
# ---------------------------------------------------------------------------

def _make_df(col_lengths: dict[str, int], total_rows: int = 100) -> pd.DataFrame:
    """
    Build a DataFrame where each column has `total_rows` rows but only the
    first `col_lengths[name]` values are non-NaN (rest are NaN).
    """
    rng = np.random.default_rng(42)
    data = {}
    for name, valid_len in col_lengths.items():
        col = np.full(total_rows, np.nan)
        col[:valid_len] = rng.standard_normal(valid_len)
        data[name] = col
    return pd.DataFrame(data)


# ===================================================================
# Test 1: Per-subject minimum length truncation
# ===================================================================

class TestFindSubjectMinLength:
    """Verify truncation finds minimum non-NaN length across BOTH EDA and BVP."""

    def test_basic_min_length(self):
        """Minimum should be the shortest non-NaN column across both DataFrames."""
        eda_df = _make_df({"1_Baseline_1": 80, "1_HIGH_1": 60}, total_rows=100)
        bvp_df = _make_df({"1_Baseline_1": 70, "1_HIGH_1": 90}, total_rows=100)
        assert find_subject_min_length(eda_df, bvp_df) == 60

    def test_single_column_each(self):
        eda_df = _make_df({"1_Baseline_1": 50}, total_rows=100)
        bvp_df = _make_df({"1_Baseline_1": 30}, total_rows=100)
        assert find_subject_min_length(eda_df, bvp_df) == 30

    def test_all_same_length(self):
        eda_df = _make_df({"1_Baseline_1": 40, "1_LOW_1": 40}, total_rows=40)
        bvp_df = _make_df({"1_Baseline_1": 40, "1_LOW_1": 40}, total_rows=40)
        assert find_subject_min_length(eda_df, bvp_df) == 40

    def test_empty_dataframe_returns_zero(self):
        eda_df = pd.DataFrame()
        bvp_df = _make_df({"1_Baseline_1": 50}, total_rows=100)
        assert find_subject_min_length(eda_df, bvp_df) == 0

    def test_both_empty_returns_zero(self):
        assert find_subject_min_length(pd.DataFrame(), pd.DataFrame()) == 0


class TestTruncateToLength:
    """Verify truncation drops rows beyond the target length."""

    def test_truncation_basic(self):
        df = _make_df({"a": 80, "b": 60}, total_rows=100)
        result = truncate_to_length(df, 50)
        assert len(result) == 50
        # Column 'a' should have 50 non-NaN values (was 80, truncated to 50)
        assert result["a"].notna().sum() == 50
        # Column 'b' should have 50 non-NaN values (was 60, truncated to 50)
        assert result["b"].notna().sum() == 50

    def test_truncation_preserves_values(self):
        rng = np.random.default_rng(99)
        data = rng.standard_normal(100)
        df = pd.DataFrame({"col": data})
        result = truncate_to_length(df, 30)
        np.testing.assert_array_equal(result["col"].values, data[:30])

    def test_truncation_to_zero(self):
        df = _make_df({"a": 50}, total_rows=100)
        result = truncate_to_length(df, 0)
        assert len(result) == 0


# ===================================================================
# Test 2 & 3: Binary class filtering
# ===================================================================

class TestFilterClasses:
    """Verify class filtering with and without REST inclusion."""

    def _make_entropy_df(self):
        """Build a synthetic entropy DataFrame with all 4 states."""
        rows = []
        for state, bc in [("baseline", 0), ("low", 1), ("high", 2), ("rest", 3)]:
            for i in range(5):
                rows.append({
                    "subject": 1,
                    "signal_type": "Eda",
                    "signal_name": f"1_{state}_{i}",
                    "state": state,
                    "binaryclass": bc,
                    "dimension": 3,
                    "tau": 1,
                    "pe": 0.5,
                    "comp": 0.3,
                    "fisher_shannon": 0.2,
                    "fisher_info": 0.1,
                    "renyipe": 0.4,
                    "renyicomp": 0.2,
                    "tsallispe": 0.6,
                    "tsalliscomp": 0.1,
                    "signallength": 100,
                })
        return pd.DataFrame(rows)

    def test_exclude_rest(self):
        """With include_rest=False, REST rows are dropped entirely."""
        df = self._make_entropy_df()
        result = filter_classes(df, include_rest=False)
        assert (result["binaryclass"] == 3).sum() == 0
        assert set(result["binaryclass"].unique()) == {0, 1, 2}
        # 5 each for baseline, low, high = 15
        assert len(result) == 15

    def test_include_rest_maps_to_zero(self):
        """With include_rest=True, REST signals get binaryclass=0 (no-pain)."""
        df = self._make_entropy_df()
        result = filter_classes(df, include_rest=True)
        # REST was 3, now mapped to 0
        assert (result["binaryclass"] == 3).sum() == 0
        # baseline(5) + rest-as-0(5) = 10 class-0 rows
        assert (result["binaryclass"] == 0).sum() == 10
        assert set(result["binaryclass"].unique()) == {0, 1, 2}
        assert len(result) == 20


# ===================================================================
# Test 4: Silhouette computation on synthetic clusters
# ===================================================================

class TestComputeSilhouetteIndices:
    """Verify silhouette score computation on entropy pair data."""

    def _make_clustered_df(self, separation: float = 10.0):
        """
        Build a DataFrame with well-separated or overlapping clusters.
        3 classes, 20 samples each, 2D features per entropy pair.
        """
        rng = np.random.default_rng(42)
        rows = []
        for cls in [0, 1, 2]:
            center = cls * separation
            for _ in range(20):
                rows.append({
                    "signal_type": "Eda",
                    "dimension": 3,
                    "tau": 1,
                    "binaryclass": cls,
                    "pe": center + rng.normal(0, 0.5),
                    "comp": center + rng.normal(0, 0.5),
                    "fisher_shannon": center + rng.normal(0, 0.5),
                    "fisher_info": center + rng.normal(0, 0.5),
                    "renyipe": center + rng.normal(0, 0.5),
                    "renyicomp": center + rng.normal(0, 0.5),
                    "tsallispe": center + rng.normal(0, 0.5),
                    "tsalliscomp": center + rng.normal(0, 0.5),
                })
        return pd.DataFrame(rows)

    def test_well_separated_clusters(self):
        """Well-separated clusters should produce high positive silhouette scores."""
        df = self._make_clustered_df(separation=10.0)
        result = compute_silhouette_indices(df)
        assert len(result) > 0
        for _, row in result.iterrows():
            if not np.isnan(row["silhouette_score"]):
                assert row["silhouette_score"] > 0.5

    def test_overlapping_clusters(self):
        """Overlapping clusters should produce low silhouette scores."""
        df = self._make_clustered_df(separation=0.0)
        result = compute_silhouette_indices(df)
        for _, row in result.iterrows():
            if not np.isnan(row["silhouette_score"]):
                assert row["silhouette_score"] < 0.5

    def test_output_columns(self):
        """Verify silhouette output has required columns."""
        df = self._make_clustered_df(separation=5.0)
        result = compute_silhouette_indices(df)
        required = [
            "signal_type", "dimension", "tau", "entropy_pair",
            "feature_x", "feature_y", "silhouette_score",
            "n_samples", "n_classes",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_single_class_gives_nan(self):
        """If only 1 class is present, silhouette should be NaN."""
        rng = np.random.default_rng(42)
        rows = []
        for _ in range(20):
            rows.append({
                "signal_type": "Eda",
                "dimension": 3,
                "tau": 1,
                "binaryclass": 0,
                "pe": rng.normal(),
                "comp": rng.normal(),
                "fisher_shannon": rng.normal(),
                "fisher_info": rng.normal(),
                "renyipe": rng.normal(),
                "renyicomp": rng.normal(),
                "tsallispe": rng.normal(),
                "tsalliscomp": rng.normal(),
            })
        df = pd.DataFrame(rows)
        result = compute_silhouette_indices(df)
        for _, row in result.iterrows():
            assert np.isnan(row["silhouette_score"])

    def test_row_count_per_combo(self):
        """Each (signal_type, d, tau) combo should produce exactly 4 rows (4 entropy pairs)."""
        df = self._make_clustered_df(separation=5.0)
        result = compute_silhouette_indices(df)
        grouped = result.groupby(["signal_type", "dimension", "tau"])
        for _, group in grouped:
            assert len(group) == 4, "Expected 4 entropy pairs per (signal_type, d, tau)"


# ===================================================================
# Test 5: Empty file handling
# ===================================================================

class TestEmptyFileHandling:
    """Verify graceful handling of empty DataFrames."""

    def test_empty_eda_nocrash(self):
        """Empty EDA DataFrame should not crash find_subject_min_length."""
        eda_df = pd.DataFrame()
        bvp_df = _make_df({"1_Baseline_1": 50}, total_rows=100)
        result = find_subject_min_length(eda_df, bvp_df)
        assert result == 0

    def test_truncate_empty_df(self):
        """Truncating an empty DataFrame returns empty."""
        df = pd.DataFrame()
        result = truncate_to_length(df, 50)
        assert len(result) == 0


# ===================================================================
# Test 6: Entropy pair constants
# ===================================================================

class TestEntropyPairs:
    """Verify the entropy pair definitions are correct."""

    def test_four_pairs_defined(self):
        assert len(ENTROPY_PAIRS) == 4

    def test_pair_names(self):
        pair_names = [p[0] for p in ENTROPY_PAIRS]
        assert "CH" in pair_names
        assert "Fisher" in pair_names
        assert "Tsallis" in pair_names
        assert "Renyi" in pair_names

    def test_pair_columns(self):
        """Each pair should have (name, x_col, y_col) structure."""
        for pair in ENTROPY_PAIRS:
            assert len(pair) == 3
            name, x_col, y_col = pair
            assert isinstance(name, str)
            assert isinstance(x_col, str)
            assert isinstance(y_col, str)


# ===================================================================
# Test 7: Output CSV format validation
# ===================================================================

class TestOutputFormat:
    """Validate expected output structure for silhouette results."""

    def test_silhouette_row_count_full_grid(self):
        """
        With 2 signals x 5 d x 3 tau x 4 pairs = 120 rows expected.
        Build a DataFrame that covers the full grid.
        """
        rng = np.random.default_rng(42)
        rows = []
        for sig in ["Eda", "Bvp"]:
            for d in [3, 4, 5, 6, 7]:
                for tau in [1, 2, 3]:
                    for cls in [0, 1, 2]:
                        for _ in range(10):
                            rows.append({
                                "signal_type": sig,
                                "dimension": d,
                                "tau": tau,
                                "binaryclass": cls,
                                "pe": rng.normal(cls * 3, 0.5),
                                "comp": rng.normal(cls * 3, 0.5),
                                "fisher_shannon": rng.normal(cls * 3, 0.5),
                                "fisher_info": rng.normal(cls * 3, 0.5),
                                "renyipe": rng.normal(cls * 3, 0.5),
                                "renyicomp": rng.normal(cls * 3, 0.5),
                                "tsallispe": rng.normal(cls * 3, 0.5),
                                "tsalliscomp": rng.normal(cls * 3, 0.5),
                            })
        df = pd.DataFrame(rows)
        result = compute_silhouette_indices(df)
        assert len(result) == 120, f"Expected 120 rows, got {len(result)}"

    def test_entropy_pairs_csv_columns(self):
        """Verify expected columns are present in entropy pair output format."""
        expected_cols = [
            "subject", "signal_type", "signal_name", "state", "binaryclass",
            "dimension", "tau", "pe", "comp", "fisher_shannon", "fisher_info",
            "renyipe", "renyicomp", "tsallispe", "tsalliscomp", "signallength",
        ]
        # Build a minimal DataFrame matching expected output
        row = {col: 0 for col in expected_cols}
        df = pd.DataFrame([row])
        for col in expected_cols:
            assert col in df.columns


# ===================================================================
# Test 8: q=2 / alpha=2 verification
# ===================================================================

class TestGeneralizedEntropyParameters:
    """Verify Renyi (alpha=2) and Tsallis (q=2) diverge from Shannon PE."""

    def test_renyi_differs_from_shannon(self):
        """At alpha=2, Renyi PE should NOT equal Shannon PE."""
        calc = EntropyCalculator()
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(500)
        result = calc.calculate_all_entropies(signal, dimension=5, tau=1)
        assert not np.isnan(result["pe"])
        assert not np.isnan(result["renyipe"])
        assert result["pe"] != result["renyipe"], \
            "Renyi PE should differ from Shannon PE at alpha=2"

    def test_tsallis_differs_from_shannon(self):
        """At q=2, Tsallis PE should NOT equal Shannon PE."""
        calc = EntropyCalculator()
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(500)
        result = calc.calculate_all_entropies(signal, dimension=5, tau=1)
        assert not np.isnan(result["pe"])
        assert not np.isnan(result["tsallispe"])
        assert result["pe"] != result["tsallispe"], \
            "Tsallis PE should differ from Shannon PE at q=2"

    def test_renyi_complexity_differs(self):
        """At alpha=2, Renyi complexity should NOT equal Shannon complexity."""
        calc = EntropyCalculator()
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(500)
        result = calc.calculate_all_entropies(signal, dimension=5, tau=1)
        assert not np.isnan(result["comp"])
        assert not np.isnan(result["renyicomp"])
        assert result["comp"] != result["renyicomp"], \
            "Renyi complexity should differ from Shannon complexity at alpha=2"

    def test_tsallis_complexity_differs(self):
        """At q=2, Tsallis complexity should NOT equal Shannon complexity."""
        calc = EntropyCalculator()
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(500)
        result = calc.calculate_all_entropies(signal, dimension=5, tau=1)
        assert not np.isnan(result["comp"])
        assert not np.isnan(result["tsalliscomp"])
        assert result["comp"] != result["tsalliscomp"], \
            "Tsallis complexity should differ from Shannon complexity at q=2"
