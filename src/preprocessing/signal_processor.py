"""
SignalPreprocessor: Signal normalization and preprocessing for feature extraction.

This module provides preprocessing capabilities including:
- Z-score normalization
- Bandpass filtering (optional)
- Signal validation and cleaning
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Optional, Tuple, Dict, Any

from ..utils.logger import SystemLogger


class SignalPreprocessor:
    """
    Preprocess physiological signals for entropy-based feature extraction.

    This class provides methods for signal normalization, filtering,
    and validation to ensure data quality before feature extraction.
    """

    def __init__(self, sampling_frequency: float = 100.0):
        """
        Initialize the SignalPreprocessor.

        Args:
            sampling_frequency: Sampling frequency of signals in Hz (default: 100 Hz)
        """
        self.fs = sampling_frequency
        self.logger = SystemLogger()
        self.logger.info(
            "SignalPreprocessor initialized",
            {"sampling_frequency": self.fs}
        )

    def z_score_normalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization to standardize the signal.

        Z-score normalization transforms the signal to have zero mean
        and unit variance, making signals comparable across subjects
        and sessions.

        Args:
            signal: Input signal array

        Returns:
            Z-score normalized signal
        """
        try:
            # Remove NaN values for calculation
            clean_signal = signal[~np.isnan(signal)]

            if len(clean_signal) == 0:
                self.logger.warning("Signal contains only NaN values")
                return signal

            # Calculate mean and standard deviation
            mean = np.mean(clean_signal)
            std = np.std(clean_signal)

            # Handle edge case of zero standard deviation
            if std == 0:
                self.logger.warning(
                    "Signal has zero standard deviation",
                    {"mean": mean, "unique_values": len(np.unique(clean_signal))}
                )
                return signal - mean  # Just center the signal

            # Apply z-score normalization
            normalized = (signal - mean) / std

            self.logger.debug(
                "Z-score normalization applied",
                {"original_mean": mean, "original_std": std}
            )

            return normalized

        except Exception as e:
            self.logger.error("Error in z-score normalization", exception=e)
            return signal

    def bandpass_filter(self,
                       signal: np.ndarray,
                       lowcut: float = 0.5,
                       highcut: float = 5.0,
                       order: int = 4) -> np.ndarray:
        """
        Apply bandpass filter to the signal.

        This is particularly useful for PPG/BVP signals to remove
        high-frequency noise and baseline drift.

        Args:
            signal: Input signal array
            lowcut: Low cutoff frequency in Hz
            highcut: High cutoff frequency in Hz
            order: Filter order

        Returns:
            Bandpass filtered signal
        """
        try:
            # Check Nyquist frequency
            nyquist = 0.5 * self.fs
            if highcut >= nyquist:
                self.logger.warning(
                    f"High cutoff {highcut} Hz exceeds Nyquist frequency {nyquist} Hz"
                )
                highcut = nyquist * 0.9

            # Design the filter
            sos = scipy_signal.butter(
                order,
                [lowcut, highcut],
                btype='band',
                fs=self.fs,
                output='sos'
            )

            # Apply the filter (forward-backward for zero phase)
            filtered = scipy_signal.sosfiltfilt(sos, signal)

            self.logger.debug(
                "Bandpass filter applied",
                {"lowcut": lowcut, "highcut": highcut, "order": order}
            )

            return filtered

        except Exception as e:
            self.logger.error("Error in bandpass filtering", exception=e)
            return signal

    def preprocess_ppg(self, signal: np.ndarray) -> np.ndarray:
        """
        Preprocess PPG/BVP signal with normalization and filtering.

        This method replicates the notebook's preprocessing approach
        for blood volume pulse signals.

        Args:
            signal: Raw PPG/BVP signal

        Returns:
            Preprocessed signal
        """
        # First normalize
        normalized = self.z_score_normalize(signal)

        # Then apply bandpass filter (0.5-5 Hz for heart rate)
        filtered = self.bandpass_filter(normalized, lowcut=0.5, highcut=5.0)

        self.logger.info("PPG preprocessing complete")
        return filtered

    def preprocess_eda(self, signal: np.ndarray) -> np.ndarray:
        """
        Preprocess EDA signal with appropriate filtering.

        EDA signals typically need different preprocessing than PPG.

        Args:
            signal: Raw EDA signal

        Returns:
            Preprocessed signal
        """
        # Z-score normalize
        normalized = self.z_score_normalize(signal)

        # Optional: Apply lowpass filter for EDA (typically < 1 Hz)
        # EDA changes are slow, so we can remove high-frequency noise
        # filtered = self.lowpass_filter(normalized, cutoff=1.0)

        self.logger.info("EDA preprocessing complete")
        return normalized

    def preprocess_resp(self, signal: np.ndarray) -> np.ndarray:
        """
        Preprocess respiration signal.

        Args:
            signal: Raw respiration signal

        Returns:
            Preprocessed signal
        """
        # Z-score normalize
        normalized = self.z_score_normalize(signal)

        # Optional: Bandpass for typical breathing rates (0.1-0.5 Hz)
        # filtered = self.bandpass_filter(normalized, lowcut=0.1, highcut=0.5)

        self.logger.info("RESP preprocessing complete")
        return normalized

    def preprocess_spo2(self, signal: np.ndarray) -> np.ndarray:
        """
        Preprocess SpO2 signal.

        SpO2 typically needs minimal preprocessing as it's already
        a percentage value.

        Args:
            signal: Raw SpO2 signal

        Returns:
            Preprocessed signal
        """
        # For SpO2, we might want to handle outliers differently
        # since valid values should be between 0-100%

        # Remove physiologically impossible values
        signal = np.where((signal < 0) | (signal > 100), np.nan, signal)

        # Z-score normalize
        normalized = self.z_score_normalize(signal)

        self.logger.info("SpO2 preprocessing complete")
        return normalized

    def validate_signal(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Validate signal quality and return statistics.

        Args:
            signal: Input signal to validate

        Returns:
            Dictionary with validation results and statistics
        """
        validation = {
            'is_valid': True,
            'length': len(signal),
            'nan_count': np.sum(np.isnan(signal)),
            'nan_percentage': 0.0,
            'unique_values': 0,
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan
        }

        try:
            # Calculate NaN percentage
            validation['nan_percentage'] = (validation['nan_count'] / len(signal)) * 100

            # Check if too many NaN values
            if validation['nan_percentage'] > 50:
                validation['is_valid'] = False
                self.logger.warning(
                    "Signal has too many NaN values",
                    {"percentage": validation['nan_percentage']}
                )

            # Get clean signal for statistics
            clean_signal = signal[~np.isnan(signal)]

            if len(clean_signal) > 0:
                validation['unique_values'] = len(np.unique(clean_signal))
                validation['mean'] = float(np.mean(clean_signal))
                validation['std'] = float(np.std(clean_signal))
                validation['min'] = float(np.min(clean_signal))
                validation['max'] = float(np.max(clean_signal))

                # Check for sufficient variability
                if validation['unique_values'] < 3:
                    validation['is_valid'] = False
                    self.logger.warning("Signal has insufficient variability")

            else:
                validation['is_valid'] = False
                self.logger.warning("Signal contains only NaN values")

        except Exception as e:
            self.logger.error("Error validating signal", exception=e)
            validation['is_valid'] = False

        return validation

    def preprocess_signal(self,
                         signal: np.ndarray,
                         signal_type: str,
                         apply_filter: bool = False) -> np.ndarray:
        """
        Preprocess signal based on its type.

        Args:
            signal: Input signal array
            signal_type: Type of signal ('Bvp', 'Eda', 'Resp', 'SpO2')
            apply_filter: Whether to apply filtering (default: False for basic z-score only)

        Returns:
            Preprocessed signal
        """
        # Validate signal first
        validation = self.validate_signal(signal)

        if not validation['is_valid']:
            self.logger.warning(
                f"Signal validation failed for {signal_type}",
                validation
            )

        # Apply preprocessing based on signal type
        signal_type_lower = signal_type.lower()

        if apply_filter:
            # With filtering (as shown in notebook)
            if signal_type_lower == 'bvp':
                return self.preprocess_ppg(signal)
            elif signal_type_lower == 'eda':
                return self.preprocess_eda(signal)
            elif signal_type_lower == 'resp':
                return self.preprocess_resp(signal)
            elif signal_type_lower == 'spo2':
                return self.preprocess_spo2(signal)
            else:
                self.logger.warning(f"Unknown signal type: {signal_type}")
                return self.z_score_normalize(signal)
        else:
            # Simple z-score normalization for all signals
            return self.z_score_normalize(signal)

    def batch_preprocess(self,
                        signals: Dict[str, np.ndarray],
                        apply_filter: bool = False) -> Dict[str, np.ndarray]:
        """
        Preprocess multiple signals in batch.

        Args:
            signals: Dictionary of signal_name -> signal_array
            apply_filter: Whether to apply filtering

        Returns:
            Dictionary of preprocessed signals
        """
        preprocessed = {}

        for name, signal in signals.items():
            # Extract signal type from name (e.g., "1_Baseline_1" -> assume from directory context)
            # This would need to be passed from the data loader
            signal_type = 'unknown'  # Will be determined from file path

            preprocessed[name] = self.preprocess_signal(
                signal,
                signal_type,
                apply_filter
            )

        self.logger.info(f"Batch preprocessing complete for {len(signals)} signals")
        return preprocessed