"""
Pain state label extraction from signal column names.

Regex-based extraction converting signal identifiers (e.g., "1_Baseline_1")
to categorical states and numeric encodings (0=baseline, 1=low, 2=high, 3=rest).
"""

import re
from typing import Tuple


class LabelExtractor:
    """
    Regex-based pain state extraction from signal identifiers.

    Maps column names (e.g., "1_Baseline_1", "6_LOW_3") to states and numeric encodings.
    """

    def __init__(self):
        """Initialize regex patterns for state matching."""
        self.baseline_pattern = re.compile(r"baseline", re.IGNORECASE)
        self.high_pattern = re.compile(r"high", re.IGNORECASE)
        self.low_pattern = re.compile(r"low", re.IGNORECASE)
        self.rest_pattern = re.compile(r"rest", re.IGNORECASE)

    def extract_state(self, signal_name: str) -> str:
        """
        Extract state label via regex matching.

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
        Convert state to numeric encoding.

        Encoding: baseline=0, low=1, high=2, rest=3, unknown=-1

        Args:
            state: Categorical state label

        Returns:
            Numeric binary class encoding

        Examples:
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
        Extract (state, binaryclass) tuple from signal name.

        Args:
            signal_name: Column name from CSV

        Returns:
            Tuple of (state, binaryclass)

        Examples:
            >>> extractor.extract_labels("1_Baseline_1")
            ('baseline', 0)
            >>> extractor.extract_labels("6_HIGH_3")
            ('high', 2)
        """
        state = self.extract_state(signal_name)
        binaryclass = self.state_to_binaryclass(state)
        return state, binaryclass
