"""
Tests for concatenated silhouette analysis pipeline.

All tests use synthetic/mock data. No real dataset access.
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from extract_concatenated_silhouette import (
    group_columns_by_condition,
    concatenate_condition_signals,
    compute_silhouette_indices,
    CONDITION_TO_BINARY,
    CONDITIONS,
)
from src.preprocessing.label_extractor import LabelExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def label_extractor():
    return LabelExtractor()


@pytest.fixture
def sample_subject_df():
    """Create a synthetic subject DataFrame mimicking real data structure.

    Columns:
      - 1_Baseline_1: 100 values (full length, no NaN)
      - 1_LOW_1, 1_LOW_2: 50 values each (padded with NaN)
      - 1_HIGH_1, 1_HIGH_2: 50 values each (padded with NaN)
      - 1_REST_1: 60 values (padded with NaN, should be excluded)
    """
    np.random.seed(42)
    n_rows = 100

    data = {}

    # Baseline: full length
    data["1_Baseline_1"] = np.random.randn(n_rows) * 0.5 + 5.0

    # Low pain: 2 columns, 50 values each
    for i, col_name in enumerate(["1_LOW_1", "1_LOW_2"]):
        values = np.full(n_rows, np.nan)
        values[:50] = np.random.randn(50) * 0.8 + 4.0
        data[col_name] = values

    # High pain: 2 columns, 50 values each
    for i, col_name in enumerate(["1_HIGH_1", "1_HIGH_2"]):
        values = np.full(n_rows, np.nan)
        values[:50] = np.random.randn(50) * 1.0 + 3.0
        data[col_name] = values

    # REST: should be excluded
    rest_values = np.full(n_rows, np.nan)
    rest_values[:60] = np.random.randn(60) * 0.3 + 5.5
    data["1_REST_1"] = rest_values

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests: Column grouping
# ---------------------------------------------------------------------------

class TestGroupColumnsByCondition:

    def test_groups_baseline_columns(self, sample_subject_df, label_extractor):
        groups = group_columns_by_condition(sample_subject_df, label_extractor)
        assert groups["baseline"] == ["1_Baseline_1"]

    def test_groups_low_columns(self, sample_subject_df, label_extractor):
        groups = group_columns_by_condition(sample_subject_df, label_extractor)
        assert sorted(groups["low"]) == ["1_LOW_1", "1_LOW_2"]

    def test_groups_high_columns(self, sample_subject_df, label_extractor):
        groups = group_columns_by_condition(sample_subject_df, label_extractor)
        assert sorted(groups["high"]) == ["1_HIGH_1", "1_HIGH_2"]

    def test_excludes_rest_columns(self, sample_subject_df, label_extractor):
        groups = group_columns_by_condition(sample_subject_df, label_extractor)
        # REST should not appear in any condition group
        all_grouped = groups["baseline"] + groups["low"] + groups["high"]
        assert "1_REST_1" not in all_grouped

    def test_empty_dataframe(self, label_extractor):
        empty_df = pd.DataFrame()
        groups = group_columns_by_condition(empty_df, label_extractor)
        assert groups == {"baseline": [], "low": [], "high": []}

    def test_case_insensitive_matching(self, label_extractor):
        df = pd.DataFrame({
            "1_BASELINE_1": [1.0, 2.0],
            "1_low_1": [3.0, 4.0],
            "1_High_1": [5.0, 6.0],
        })
        groups = group_columns_by_condition(df, label_extractor)
        assert len(groups["baseline"]) == 1
        assert len(groups["low"]) == 1
        assert len(groups["high"]) == 1


# ---------------------------------------------------------------------------
# Tests: Signal concatenation
# ---------------------------------------------------------------------------

class TestConcatenateConditionSignals:

    def test_concatenation_length(self, sample_subject_df):
        """Concatenated low-pain should be 50 + 50 = 100 non-NaN values."""
        columns = ["1_LOW_1", "1_LOW_2"]
        result = concatenate_condition_signals(sample_subject_df, columns)
        assert len(result) == 100  # 50 + 50

    def test_concatenation_baseline_full_length(self, sample_subject_df):
        """Baseline has all 100 rows non-NaN."""
        result = concatenate_condition_signals(sample_subject_df, ["1_Baseline_1"])
        assert len(result) == 100

    def test_concatenation_strips_nans(self):
        """NaN values should be stripped before concatenation."""
        df = pd.DataFrame({
            "col1": [1.0, 2.0, np.nan, np.nan],
            "col2": [np.nan, 3.0, 4.0, np.nan],
        })
        result = concatenate_condition_signals(df, ["col1", "col2"])
        assert len(result) == 4  # 2 from col1 + 2 from col2
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])

    def test_concatenation_preserves_order(self):
        """Values should appear in column order, then row order within column."""
        df = pd.DataFrame({
            "a": [10.0, 20.0],
            "b": [30.0, 40.0],
        })
        result = concatenate_condition_signals(df, ["a", "b"])
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0, 40.0])

    def test_empty_columns(self):
        """All-NaN columns should contribute nothing."""
        df = pd.DataFrame({
            "col1": [np.nan, np.nan],
            "col2": [1.0, 2.0],
        })
        result = concatenate_condition_signals(df, ["col1", "col2"])
        assert len(result) == 2
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_no_columns(self):
        """Empty column list returns empty array."""
        df = pd.DataFrame({"a": [1.0]})
        result = concatenate_condition_signals(df, [])
        assert len(result) == 0

    def test_all_nan_columns(self):
        """All-NaN columns return empty array."""
        df = pd.DataFrame({
            "col1": [np.nan, np.nan],
            "col2": [np.nan, np.nan],
        })
        result = concatenate_condition_signals(df, ["col1", "col2"])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: Binary label mapping
# ---------------------------------------------------------------------------

class TestBinaryLabelMapping:

    def test_baseline_is_nopain(self):
        assert CONDITION_TO_BINARY["baseline"] == 0

    def test_low_is_pain(self):
        assert CONDITION_TO_BINARY["low"] == 1

    def test_high_is_pain(self):
        assert CONDITION_TO_BINARY["high"] == 1

    def test_all_conditions_covered(self):
        """Every condition in CONDITIONS has a binary mapping."""
        for cond in CONDITIONS:
            assert cond in CONDITION_TO_BINARY


# ---------------------------------------------------------------------------
# Tests: Silhouette computation
# ---------------------------------------------------------------------------

class TestComputeSilhouetteIndices:

    def test_output_columns(self):
        """Silhouette output should have expected columns."""
        entropy_df = pd.DataFrame({
            "signal_type": ["Eda"] * 6,
            "dimension": [3] * 6,
            "tau": [1] * 6,
            "binaryclass": [0, 0, 0, 1, 1, 1],
            "pe": [0.1, 0.15, 0.12, 0.8, 0.85, 0.82],
            "comp": [0.9, 0.85, 0.88, 0.2, 0.15, 0.18],
            "fisher_shannon": [0.1, 0.15, 0.12, 0.8, 0.85, 0.82],
            "fisher_info": [0.9, 0.85, 0.88, 0.2, 0.15, 0.18],
            "tsallispe": [0.1, 0.15, 0.12, 0.8, 0.85, 0.82],
            "tsalliscomp": [0.9, 0.85, 0.88, 0.2, 0.15, 0.18],
            "renyipe": [0.1, 0.15, 0.12, 0.8, 0.85, 0.82],
            "renyicomp": [0.9, 0.85, 0.88, 0.2, 0.15, 0.18],
        })

        result = compute_silhouette_indices(entropy_df)

        expected_cols = {
            "signal_type", "dimension", "tau", "entropy_pair",
            "feature_x", "feature_y", "silhouette_score",
            "n_samples", "n_classes",
        }
        assert set(result.columns) == expected_cols

    def test_silhouette_range(self):
        """Silhouette scores should be in [-1, 1]."""
        np.random.seed(42)
        n = 30
        entropy_df = pd.DataFrame({
            "signal_type": ["Eda"] * n,
            "dimension": [3] * n,
            "tau": [1] * n,
            "binaryclass": [0] * 10 + [1] * 20,
            "pe": np.random.rand(n),
            "comp": np.random.rand(n),
            "fisher_shannon": np.random.rand(n),
            "fisher_info": np.random.rand(n),
            "tsallispe": np.random.rand(n),
            "tsalliscomp": np.random.rand(n),
            "renyipe": np.random.rand(n),
            "renyicomp": np.random.rand(n),
        })

        result = compute_silhouette_indices(entropy_df)
        valid = result.dropna(subset=["silhouette_score"])

        for _, row in valid.iterrows():
            assert -1.0 <= row["silhouette_score"] <= 1.0

    def test_well_separated_clusters(self):
        """Well-separated binary clusters should produce high silhouette."""
        np.random.seed(42)
        n = 20
        # Cluster 0 centered at (0.1, 0.9), cluster 1 at (0.9, 0.1)
        pe_0 = np.random.normal(0.1, 0.01, n)
        comp_0 = np.random.normal(0.9, 0.01, n)
        pe_1 = np.random.normal(0.9, 0.01, n)
        comp_1 = np.random.normal(0.1, 0.01, n)

        entropy_df = pd.DataFrame({
            "signal_type": ["Eda"] * (2 * n),
            "dimension": [3] * (2 * n),
            "tau": [1] * (2 * n),
            "binaryclass": [0] * n + [1] * n,
            "pe": np.concatenate([pe_0, pe_1]),
            "comp": np.concatenate([comp_0, comp_1]),
            "fisher_shannon": np.concatenate([pe_0, pe_1]),
            "fisher_info": np.concatenate([comp_0, comp_1]),
            "tsallispe": np.concatenate([pe_0, pe_1]),
            "tsalliscomp": np.concatenate([comp_0, comp_1]),
            "renyipe": np.concatenate([pe_0, pe_1]),
            "renyicomp": np.concatenate([comp_0, comp_1]),
        })

        result = compute_silhouette_indices(entropy_df)
        ch_score = result[result["entropy_pair"] == "CH"]["silhouette_score"].iloc[0]
        assert ch_score > 0.8

    def test_four_entropy_pairs_per_group(self):
        """Each (signal, d, tau) group should produce 4 silhouette rows."""
        entropy_df = pd.DataFrame({
            "signal_type": ["Eda"] * 6,
            "dimension": [3] * 6,
            "tau": [1] * 6,
            "binaryclass": [0, 0, 0, 1, 1, 1],
            "pe": [0.1, 0.15, 0.12, 0.8, 0.85, 0.82],
            "comp": [0.9, 0.85, 0.88, 0.2, 0.15, 0.18],
            "fisher_shannon": [0.1, 0.15, 0.12, 0.8, 0.85, 0.82],
            "fisher_info": [0.9, 0.85, 0.88, 0.2, 0.15, 0.18],
            "tsallispe": [0.1, 0.15, 0.12, 0.8, 0.85, 0.82],
            "tsalliscomp": [0.9, 0.85, 0.88, 0.2, 0.15, 0.18],
            "renyipe": [0.1, 0.15, 0.12, 0.8, 0.85, 0.82],
            "renyicomp": [0.9, 0.85, 0.88, 0.2, 0.15, 0.18],
        })

        result = compute_silhouette_indices(entropy_df)
        assert len(result) == 4  # CH, Fisher, Tsallis, Renyi

    def test_handles_single_class(self):
        """Single-class data should produce NaN silhouette."""
        entropy_df = pd.DataFrame({
            "signal_type": ["Eda"] * 3,
            "dimension": [3] * 3,
            "tau": [1] * 3,
            "binaryclass": [0, 0, 0],  # All same class
            "pe": [0.1, 0.15, 0.12],
            "comp": [0.9, 0.85, 0.88],
            "fisher_shannon": [0.1, 0.15, 0.12],
            "fisher_info": [0.9, 0.85, 0.88],
            "tsallispe": [0.1, 0.15, 0.12],
            "tsalliscomp": [0.9, 0.85, 0.88],
            "renyipe": [0.1, 0.15, 0.12],
            "renyicomp": [0.9, 0.85, 0.88],
        })

        result = compute_silhouette_indices(entropy_df)
        assert all(np.isnan(result["silhouette_score"]))


# ---------------------------------------------------------------------------
# Tests: End-to-end integration (with mocked entropy calculator)
# ---------------------------------------------------------------------------

class TestEndToEndConcatenation:

    def test_full_subject_produces_three_conditions(self, sample_subject_df, label_extractor):
        """A complete subject should produce data for all 3 conditions."""
        groups = group_columns_by_condition(sample_subject_df, label_extractor)

        for condition in CONDITIONS:
            assert len(groups[condition]) > 0, f"Missing {condition} columns"

    def test_concatenated_lengths_match_expectations(self, sample_subject_df, label_extractor):
        """Concatenated signal lengths should sum individual column lengths."""
        groups = group_columns_by_condition(sample_subject_df, label_extractor)

        for condition in CONDITIONS:
            columns = groups[condition]
            concat = concatenate_condition_signals(sample_subject_df, columns)

            # Sum of non-NaN values across all columns of this condition
            expected_length = sum(
                sample_subject_df[col].notna().sum() for col in columns
            )
            assert len(concat) == expected_length

    def test_rest_excluded_from_all_conditions(self, sample_subject_df, label_extractor):
        """REST columns should never appear in any condition group."""
        groups = group_columns_by_condition(sample_subject_df, label_extractor)

        for condition in CONDITIONS:
            for col in groups[condition]:
                state = label_extractor.extract_state(col)
                assert state != "rest"
