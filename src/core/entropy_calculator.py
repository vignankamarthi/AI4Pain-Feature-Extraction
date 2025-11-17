"""
Ordpy-based entropy calculator for physiological signals.

Implements 8 information-theoretic measures across parametrized temporal scales
(embedding dimension d, time delay tau). Wraps ordpy with validation, NaN handling,
and structured logging.
"""

import numpy as np
import ordpy
import warnings
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass

from ..utils.logger import SystemLogger


@dataclass
class EntropyFeatures:
    """8-measure entropy feature vector for single (d, tau) combination."""
    pe: float
    comp: float
    fisher_shannon: float
    fisher_info: float
    renyipe: float
    renyicomp: float
    tsallispe: float
    tsalliscomp: float
    dimension: int
    tau: int


class EntropyCalculator:
    """
    Ordpy wrapper for batch entropy extraction with validation.

    Handles edge cases: insufficient signal length, low unique value count, NaN/Inf results.
    Returns NaN-filled dicts on failure for downstream robustness.
    """

    def __init__(self):
        """Initialize with structured logger."""
        self.logger = SystemLogger()
        self.logger.info("EntropyCalculator initialized")

    def calculate_all_entropies(self,
                               signal: np.ndarray,
                               dimension: int,
                               tau: int) -> Dict[str, float]:
        """
        Compute all 8 entropy measures for (signal, d, tau) triplet.

        Returns NaN-filled dict on failure (signal too short, insufficient unique values,
        ordpy exceptions). Never raises - critical for pipeline robustness.

        Args:
            signal: Input time series data
            dimension: Embedding dimension (dx parameter in ordpy)
            tau: Time delay (taux parameter in ordpy)

        Returns:
            Dictionary containing all entropy measures
        """
        # Default to NaN for failed calculations
        result = {
            'pe': np.nan,
            'comp': np.nan,
            'fisher_shannon': np.nan,
            'fisher_info': np.nan,
            'renyipe': np.nan,
            'renyicomp': np.nan,
            'tsallispe': np.nan,
            'tsalliscomp': np.nan,
            'dimension': dimension,
            'tau': tau,
            'signallength': len(signal)
        }

        try:
            clean_signal = signal[~np.isnan(signal)]

            # Validate minimum length for ordpy permutation patterns
            if len(clean_signal) < dimension * tau:
                self.logger.warning(
                    f"Signal too short for d={dimension}, tau={tau}",
                    {"signal_length": len(clean_signal), "required": dimension * tau}
                )
                return result

            # Ordpy requires at least d unique values for permutation patterns
            unique_values = np.unique(clean_signal)
            if len(unique_values) < dimension:
                self.logger.warning(
                    f"Insufficient unique values for dimension {dimension}",
                    {"unique_values": len(unique_values), "required": dimension}
                )
                return result

            with warnings.catch_warnings():
                warnings.simplefilter("always")

                # Ordpy tuple unpacking: (entropy, complexity) pairs
                ans1 = ordpy.complexity_entropy(clean_signal, dx=dimension, taux=tau)
                pe = ans1[0]
                comp = ans1[1]

                ans2 = ordpy.fisher_shannon(clean_signal, dx=dimension, taux=tau)
                fisher_shannon = ans2[0]
                fisher_info = ans2[1]

                ans3 = ordpy.renyi_complexity_entropy(clean_signal, dx=dimension, taux=tau)
                renyipe = ans3[0]
                renyicomp = ans3[1]

                ans4 = ordpy.tsallis_complexity_entropy(clean_signal, dx=dimension, taux=tau)
                tsallispe = ans4[0]
                tsalliscomp = ans4[1]

            # NaN/Inf sanitization before returning
            result['pe'] = self._validate_value(pe, 'PE')
            result['comp'] = self._validate_value(comp, 'Complexity')
            result['fisher_shannon'] = self._validate_value(fisher_shannon, 'Fisher-Shannon')
            result['fisher_info'] = self._validate_value(fisher_info, 'Fisher Information')
            result['renyipe'] = self._validate_value(renyipe, 'Renyi PE')
            result['renyicomp'] = self._validate_value(renyicomp, 'Renyi Complexity')
            result['tsallispe'] = self._validate_value(tsallispe, 'Tsallis PE')
            result['tsalliscomp'] = self._validate_value(tsalliscomp, 'Tsallis Complexity')

            self.logger.debug(
                f"Entropy calculation successful",
                {"dimension": dimension, "tau": tau, "pe": result['pe']}
            )

        except MemoryError as e:
            self.logger.error(
                f"Memory error during entropy calculation",
                exception=e,
                context={"dimension": dimension, "tau": tau}
            )
        except Exception as e:
            self.logger.error(
                f"Error calculating entropies",
                exception=e,
                context={"dimension": dimension, "tau": tau}
            )

        return result

    def _validate_value(self, value: float, name: str) -> float:
        """
        Sanitize NaN/Inf from ordpy outputs (edge case: numerical instability).

        Args:
            value: The calculated value
            name: Name of the measure for logging

        Returns:
            Validated value (NaN if invalid)
        """
        if np.isnan(value) or np.isinf(value):
            self.logger.warning(f"Invalid {name} value: {value}")
            return np.nan
        return value

    def calculate_permutation_entropy(self,
                                     signal: np.ndarray,
                                     dimension: int,
                                     tau: int) -> float:
        """
        Extract PE only (for single-measure use cases).

        Args:
            signal: Input time series
            dimension: Embedding dimension
            tau: Time delay

        Returns:
            Permutation entropy value
        """
        try:
            clean_signal = signal[~np.isnan(signal)]
            result = ordpy.complexity_entropy(clean_signal, dx=dimension, taux=tau)
            return self._validate_value(result[0], 'PE')
        except Exception as e:
            self.logger.error("Error calculating PE", exception=e)
            return np.nan

    def calculate_complexity(self,
                           signal: np.ndarray,
                           dimension: int,
                           tau: int) -> float:
        """
        Calculate statistical complexity.

        Args:
            signal: Input time series
            dimension: Embedding dimension
            tau: Time delay

        Returns:
            Statistical complexity value
        """
        try:
            clean_signal = signal[~np.isnan(signal)]
            result = ordpy.complexity_entropy(clean_signal, dx=dimension, taux=tau)
            return self._validate_value(result[1], 'Complexity')
        except Exception as e:
            self.logger.error("Error calculating complexity", exception=e)
            return np.nan

    def calculate_fisher_information(self,
                                    signal: np.ndarray,
                                    dimension: int,
                                    tau: int) -> float:
        """
        Calculate Fisher information measure.

        Args:
            signal: Input time series
            dimension: Embedding dimension
            tau: Time delay

        Returns:
            Fisher information value
        """
        try:
            clean_signal = signal[~np.isnan(signal)]
            result = ordpy.fisher_shannon(clean_signal, dx=dimension, taux=tau)
            return self._validate_value(result[1], 'Fisher')
        except Exception as e:
            self.logger.error("Error calculating Fisher information", exception=e)
            return np.nan

    def calculate_renyi_entropy(self,
                               signal: np.ndarray,
                               dimension: int,
                               tau: int) -> Tuple[float, float]:
        """
        Calculate Renyi entropy and complexity.

        Args:
            signal: Input time series
            dimension: Embedding dimension
            tau: Time delay

        Returns:
            Tuple of (Renyi entropy, Renyi complexity)
        """
        try:
            clean_signal = signal[~np.isnan(signal)]
            result = ordpy.renyi_complexity_entropy(clean_signal, dx=dimension, taux=tau)
            return (
                self._validate_value(result[0], 'Renyi PE'),
                self._validate_value(result[1], 'Renyi Complexity')
            )
        except Exception as e:
            self.logger.error("Error calculating Renyi entropy", exception=e)
            return (np.nan, np.nan)

    def calculate_tsallis_entropy(self,
                                 signal: np.ndarray,
                                 dimension: int,
                                 tau: int) -> Tuple[float, float]:
        """
        Calculate Tsallis entropy and complexity.

        Args:
            signal: Input time series
            dimension: Embedding dimension
            tau: Time delay

        Returns:
            Tuple of (Tsallis entropy, Tsallis complexity)
        """
        try:
            clean_signal = signal[~np.isnan(signal)]
            result = ordpy.tsallis_complexity_entropy(clean_signal, dx=dimension, taux=tau)
            return (
                self._validate_value(result[0], 'Tsallis PE'),
                self._validate_value(result[1], 'Tsallis Complexity')
            )
        except Exception as e:
            self.logger.error("Error calculating Tsallis entropy", exception=e)
            return (np.nan, np.nan)

    def batch_calculate(self,
                       signal: np.ndarray,
                       dimensions: List[int] = [3, 4, 5, 6, 7],
                       taus: List[int] = [1, 2, 3]) -> List[Dict[str, float]]:
        """
        Compute entropy grid: len(dimensions) × len(taus) parameter sweep.

        Returns list of dicts (long-format output). Processes lower dimensions first
        (faster execution, less memory). Forces GC after each to prevent accumulation.

        Args:
            signal: Input time series
            dimensions: List of embedding dimensions to try
            taus: List of time delays to try

        Returns:
            List of entropy results for each combination
        """
        results = []
        total_combinations = len(dimensions) * len(taus)

        self.logger.info(
            f"Starting batch entropy calculation",
            {"combinations": total_combinations, "dimensions": dimensions, "taus": taus}
        )

        for dim in sorted(dimensions):
            for tau in taus:
                start_time = self.logger.start_operation(f"Entropy d={dim}, tau={tau}")

                result = self.calculate_all_entropies(signal, dim, tau)
                results.append(result)

                self.logger.end_operation(f"Entropy d={dim}, tau={tau}", start_time)

                import gc
                gc.collect()

        self.logger.info(f"Batch calculation complete", {"results": len(results)})
        return results