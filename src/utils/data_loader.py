"""
DataLoader: Robust CSV data loading with defensive coding.

This module handles loading physiological signals from CSV files
with comprehensive error handling and validation.
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .logger import SystemLogger


class DataLoader:
    """
    Load and organize physiological signal data from CSV files.

    Features:
    - Defensive file reading with error handling
    - Case-insensitive state extraction
    - Automatic signal type detection
    - Progress tracking for large datasets
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DataLoader.

        Args:
            data_dir: Base directory containing train/test subdirectories
        """
        self.data_dir = Path(data_dir)
        self.logger = SystemLogger()
        self.logger.info(f"DataLoader initialized with data directory: {self.data_dir}")

        # Define expected signal types
        self.signal_types = ['Bvp', 'Eda', 'Resp', 'SpO2']

    def numerical_sort(self, filename: str) -> int:
        """
        Extract numerical value for sorting files.

        Replicates the notebook's sorting approach.

        Args:
            filename: Name of the file

        Returns:
            Numerical value for sorting
        """
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else -1

    def extract_state(self, signal_name: str) -> str:
        """
        Extract pain state from signal name with case-insensitive matching.

        Args:
            signal_name: Signal column name (e.g., "1_Baseline_1", "63_HIGH_12")

        Returns:
            Extracted state ('baseline', 'low', 'high', 'rest', or 'unknown')
        """
        try:
            # Convert to lowercase for case-insensitive matching
            signal_lower = signal_name.lower()

            # Check for each state
            if 'baseline' in signal_lower:
                return 'baseline'
            elif 'high' in signal_lower:
                return 'high'
            elif 'low' in signal_lower:
                return 'low'
            elif 'rest' in signal_lower:
                return 'rest'
            else:
                # Try to extract from underscore-separated format
                parts = signal_name.split('_')
                if len(parts) >= 2:
                    state = parts[1].lower()
                    if state in ['baseline', 'high', 'low', 'rest']:
                        return state

                self.logger.debug(f"Could not extract state from: {signal_name}")
                return 'unknown'

        except Exception as e:
            self.logger.error(
                f"Error extracting state from signal name",
                exception=e,
                context={"signal_name": signal_name}
            )
            return 'unknown'

    def get_binary_class(self, state: str) -> int:
        """
        Convert state to binary class label.

        Mapping (from notebook):
        - baseline = 0
        - low = 1
        - high = 2
        - rest = 3
        - unknown = -1

        Args:
            state: Pain state string

        Returns:
            Binary class label
        """
        mapping = {
            'baseline': 0,
            'low': 1,
            'high': 2,
            'rest': 3,
            'unknown': -1
        }
        return mapping.get(state, -1)

    def load_csv_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load a single CSV file with error handling.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame or None if loading fails
        """
        try:
            if not file_path.exists():
                self.logger.error(f"File not found: {file_path}")
                return None

            df = pd.read_csv(file_path)

            self.logger.debug(
                f"Loaded CSV file",
                {"file": file_path.name, "shape": df.shape, "columns": len(df.columns)}
            )

            return df

        except pd.errors.EmptyDataError:
            self.logger.error(f"Empty CSV file: {file_path}")
            return None
        except Exception as e:
            self.logger.error(
                f"Error loading CSV file",
                exception=e,
                context={"file": str(file_path)}
            )
            return None

    def load_directory(self,
                      directory: Path,
                      signal_type: str) -> List[Dict[str, Any]]:
        """
        Load all CSV files from a directory.

        Args:
            directory: Directory containing CSV files
            signal_type: Type of signal (Bvp, Eda, Resp, SpO2)

        Returns:
            List of dictionaries containing signal data and metadata
        """
        data_list = []

        if not directory.exists():
            self.logger.warning(f"Directory not found: {directory}")
            return data_list

        # Get all CSV files and sort them numerically
        csv_files = sorted(
            [f for f in directory.iterdir() if f.suffix == '.csv'],
            key=lambda x: self.numerical_sort(x.name)
        )

        self.logger.info(
            f"Found CSV files in {signal_type} directory",
            {"count": len(csv_files), "directory": str(directory)}
        )

        for file_path in csv_files:
            df = self.load_csv_file(file_path)

            if df is None:
                continue

            # Process each signal column in the file
            for column in df.columns:
                if not column.startswith(file_path.stem):
                    continue

                try:
                    # Extract signal data
                    signal_data = df[column].values

                    # Extract state and binary class
                    state = self.extract_state(column)
                    binary_class = self.get_binary_class(state)

                    # Create data dictionary
                    data_dict = {
                        'file_name': file_path.name,
                        'signal': column,
                        'signal_type': signal_type,
                        'data': signal_data,
                        'state': state,
                        'binaryclass': binary_class,
                        'signallength': len(signal_data)
                    }

                    data_list.append(data_dict)

                except Exception as e:
                    self.logger.error(
                        f"Error processing signal column",
                        exception=e,
                        context={"file": file_path.name, "column": column}
                    )

        self.logger.info(
            f"Loaded {len(data_list)} signals from {signal_type}",
            {"directory": str(directory)}
        )

        return data_list

    def load_all_signals(self,
                        dataset: str = 'train',
                        signal_types: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """
        Load all signals from specified dataset.

        Args:
            dataset: 'train', 'validation', or 'test'
            signal_types: List of signal types to load (default: all)

        Returns:
            Dictionary mapping signal type to list of signal data
        """
        if signal_types is None:
            signal_types = self.signal_types

        all_data = {}
        base_dir = self.data_dir / dataset

        if not base_dir.exists():
            self.logger.error(f"Dataset directory not found: {base_dir}")
            return all_data

        for signal_type in signal_types:
            signal_dir = base_dir / signal_type
            all_data[signal_type] = self.load_directory(signal_dir, signal_type)

        # Log summary statistics
        total_signals = sum(len(signals) for signals in all_data.values())
        self.logger.info(
            f"Loaded all signals from {dataset} dataset",
            {
                "total_signals": total_signals,
                "signal_types": list(all_data.keys()),
                "signals_per_type": {k: len(v) for k, v in all_data.items()}
            }
        )

        return all_data

    def create_concatenated_dataset(self,
                                   all_data: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Create a concatenated dataset from all signal types.

        Args:
            all_data: Dictionary of signal data by type

        Returns:
            Concatenated list of all signals
        """
        concatenated = []

        for signal_type, signals in all_data.items():
            concatenated.extend(signals)

        self.logger.info(
            f"Created concatenated dataset",
            {"total_signals": len(concatenated)}
        )

        return concatenated

    def get_signal_statistics(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic statistics for a signal.

        Args:
            signal_data: Signal array

        Returns:
            Dictionary of statistics
        """
        clean_signal = signal_data[~np.isnan(signal_data)]

        if len(clean_signal) == 0:
            return {
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'valid_samples': 0
            }

        return {
            'mean': float(np.mean(clean_signal)),
            'std': float(np.std(clean_signal)),
            'min': float(np.min(clean_signal)),
            'max': float(np.max(clean_signal)),
            'valid_samples': len(clean_signal)
        }

    def validate_data_structure(self) -> Dict[str, Any]:
        """
        Validate the expected data directory structure.

        Returns:
            Validation results dictionary
        """
        validation = {
            'is_valid': True,
            'missing_directories': [],
            'file_counts': {}
        }

        # Check for train, validation, and test directories
        for dataset in ['train', 'validation', 'test']:
            dataset_dir = self.data_dir / dataset

            if not dataset_dir.exists():
                validation['missing_directories'].append(str(dataset_dir))
                validation['is_valid'] = False
                continue

            # Check for signal type subdirectories
            for signal_type in self.signal_types:
                signal_dir = dataset_dir / signal_type

                if not signal_dir.exists():
                    validation['missing_directories'].append(str(signal_dir))
                    validation['is_valid'] = False
                else:
                    # Count CSV files
                    csv_count = len(list(signal_dir.glob('*.csv')))
                    key = f"{dataset}/{signal_type}"
                    validation['file_counts'][key] = csv_count

        if validation['missing_directories']:
            self.logger.warning(
                "Missing required directories",
                {"directories": validation['missing_directories']}
            )

        return validation