"""
Label Extractor: Extract pain state labels from signal column names.

This module implements the exact label extraction logic from the reference notebook,
converting signal identifiers into categorical (state) and numeric (binaryclass) labels.
"""

import re
from typing import Tuple


class LabelExtractor:
    """
    Extract pain state labels from physiological signal column names.

    Implements notebook logic for converting column identifiers like "1_Baseline_1",
    "6_LOW_3", "12_HIGH_5" into categorical state labels and numeric binary classifications.
    """

    def __init__(self):
        """Initialize the LabelExtractor with regex patterns."""
        self.baseline_pattern = re.compile(r"baseline", re.IGNORECASE)
        self.high_pattern = re.compile(r"high", re.IGNORECASE)
        self.low_pattern = re.compile(r"low", re.IGNORECASE)
        self.rest_pattern = re.compile(r"rest", re.IGNORECASE)

    def extract_state(self, signal_name: str) -> str:
        """
        Extract state label from signal column name.

        Uses regex matching to identify pain state from column identifier.
        Exact replication of notebook extract_state() function.

        Args:
            signal_name: Column name from CSV (e.g., "1_Baseline_1", "6_HIGH_3")

        Returns:
            State label: "baseline", "low", "high", "rest", or "unknown"

        Examples:
            >>> extractor = LabelExtractor()
            >>> extractor.extract_state("1_Baseline_1")
            'baseline'
            >>> extractor.extract_state("6_HIGH_3")
            'high'
            >>> extractor.extract_state("42_LOW_2")
            'low'
        """
        signal_str = str(signal_name)

        if self.baseline_pattern.search(signal_str):
            return "baseline"
        elif self.high_pattern.search(signal_str):
            return "high"
        elif self.low_pattern.search(signal_str):
            return "low"
        elif self.rest_pattern.search(signal_str):
            return "rest"
        else:
            return "unknown"

    def state_to_binaryclass(self, state: str) -> int:
        """
        Convert state label to numeric binary classification.

        Implements notebook encoding:
        - baseline -> 0
        - low -> 1
        - high -> 2
        - rest -> 3
        - unknown -> -1

        Args:
            state: Categorical state label

        Returns:
            Numeric binary class encoding

        Examples:
            >>> extractor = LabelExtractor()
            >>> extractor.state_to_binaryclass("baseline")
            0
            >>> extractor.state_to_binaryclass("high")
            2
        """
        state_mapping = {
            "baseline": 0,
            "low": 1,
            "high": 2,
            "rest": 3,
            "unknown": -1
        }
        return state_mapping.get(state, -1)

    def extract_labels(self, signal_name: str) -> Tuple[str, int]:
        """
        Extract both state and binaryclass labels from signal name.

        Convenience method combining state extraction and numeric encoding.

        Args:
            signal_name: Column name from CSV

        Returns:
            Tuple of (state, binaryclass)

        Examples:
            >>> extractor = LabelExtractor()
            >>> extractor.extract_labels("1_Baseline_1")
            ('baseline', 0)
            >>> extractor.extract_labels("6_HIGH_3")
            ('high', 2)
        """
        state = self.extract_state(signal_name)
        binaryclass = self.state_to_binaryclass(state)
        return state, binaryclass
