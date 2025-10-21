"""
FeatureExtractor: Main orchestrator for entropy-based feature extraction.

This module coordinates the entire feature extraction pipeline, integrating
data loading, preprocessing, entropy calculation, and result generation.
"""

import pandas as pd
import numpy as np
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import time

from .entropy_calculator import EntropyCalculator
from ..preprocessing.signal_processor import SignalPreprocessor
from ..preprocessing.label_extractor import LabelExtractor
from ..utils.data_loader import DataLoader
from ..utils.logger import SystemLogger
from ..config.settings import Settings


class FeatureExtractor:
    """
    Main class for extracting entropy-based features from physiological signals.

    This class orchestrates the entire pipeline from data loading to
    feature table generation, exactly matching the notebook workflow.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the FeatureExtractor.

        Args:
            settings: Configuration settings (uses defaults if None)
        """
        self.settings = settings or Settings()
        self.logger = SystemLogger()
        self.entropy_calculator = EntropyCalculator()
        self.preprocessor = SignalPreprocessor(self.settings.sampling_frequency)
        self.label_extractor = LabelExtractor()
        self.data_loader = DataLoader(self.settings.data_dir)

        self.logger.info(
            "FeatureExtractor initialized",
            self.settings.to_dict()
        )

        # Validate paths
        self.settings.validate_paths()

    def process_single_signal(self,
                             signal_data: np.ndarray,
                             file_path: str,
                             signal_name: str) -> List[Dict[str, Any]]:
        """
        Process a single signal with all dimension-tau combinations.

        Generates long-format output: each row = one signal × dimension × tau combination.

        Args:
            signal_data: Raw signal array (may contain NaN values)
            file_path: Full path to source CSV file
            signal_name: Column name from CSV (e.g., "1_Baseline_1")

        Returns:
            List of feature dictionaries (15 rows per signal: 5 dimensions × 3 taus)
        """
        results = []

        # Calculate NaN percentage before preprocessing
        total_length = len(signal_data)
        nan_count = np.sum(np.isnan(signal_data))
        nan_percentage = (nan_count / total_length * 100) if total_length > 0 else 0.0

        # Extract labels
        state, binaryclass = self.label_extractor.extract_labels(signal_name)

        # Remove NaN values for entropy calculation
        clean_signal = signal_data[~np.isnan(signal_data)]
        signal_length = len(clean_signal)

        # Apply preprocessing if enabled
        if self.settings.apply_z_score and signal_length > 0:
            clean_signal = self.preprocessor.z_score_normalize(clean_signal)

        # Calculate entropies for each dimension-tau combination
        for dim in self.settings.dimensions:
            for tau in self.settings.taus:
                # Calculate all entropy measures
                entropy_dict = self.entropy_calculator.calculate_all_entropies(
                    clean_signal, dim, tau
                )

                # Build long-format row matching notebook structure (16 columns)
                result = {
                    'file_name': file_path,
                    'signal': signal_name,
                    'signallength': signal_length,
                    'pe': entropy_dict['pe'],
                    'comp': entropy_dict['comp'],
                    'fisher_shannon': entropy_dict['fisher_shannon'],
                    'fisher_info': entropy_dict['fisher_info'],
                    'renyipe': entropy_dict['renyipe'],
                    'renyicomp': entropy_dict['renyicomp'],
                    'tsallispe': entropy_dict['tsallispe'],
                    'tsalliscomp': entropy_dict['tsalliscomp'],
                    'dimension': dim,
                    'tau': tau,
                    'state': state,
                    'binaryclass': binaryclass,
                    'nan_percentage': round(nan_percentage, 2)
                }

                results.append(result)

                # Force garbage collection
                gc.collect()

        return results

    def process_signal_type(self,
                           dataset: str,
                           signal_type: str) -> pd.DataFrame:
        """
        Process all signals of a specific type and generate long-format output.

        Each CSV file contains multiple signal columns. Each signal generates 15 rows
        (5 dimensions × 3 taus) in the output DataFrame.

        Args:
            dataset: 'train', 'validation', or 'test'
            signal_type: Type of signal (Bvp, Eda, Resp, SpO2)

        Returns:
            DataFrame with long-format features (16 columns per row)
        """
        self.logger.info(f"Processing {signal_type} signals from {dataset}")

        # Get signal directory
        signal_dir = Path(self.settings.data_dir) / dataset / signal_type

        if not signal_dir.exists():
            self.logger.warning(f"Directory not found: {signal_dir}")
            return pd.DataFrame()

        # Get all CSV files and sort them numerically
        csv_files = sorted(
            [f for f in signal_dir.iterdir() if f.suffix == '.csv'],
            key=lambda x: self.data_loader.numerical_sort(x.name)
        )

        if not csv_files:
            self.logger.warning(f"No CSV files found in {signal_dir}")
            return pd.DataFrame()

        self.logger.info(f"Found {len(csv_files)} CSV files for {signal_type}")

        all_results = []

        # Process each CSV file
        for csv_file in tqdm(csv_files, desc=f"Processing {signal_type}", unit="file",
                            disable=not self.settings.show_progress):
            try:
                # Load CSV file
                df = self.data_loader.load_csv_file(csv_file)
                if df is None:
                    continue

                # Process each signal column in the CSV
                for column_name in df.columns:
                    try:
                        signal_data = df[column_name].values

                        # Build full path for file_name column
                        relative_path = csv_file.relative_to(Path(self.settings.data_dir))
                        file_path = f"data/{relative_path}"

                        # Process signal and get 15 rows (5 dims × 3 taus)
                        signal_results = self.process_single_signal(
                            signal_data,
                            file_path,
                            column_name
                        )

                        all_results.extend(signal_results)

                    except Exception as e:
                        self.logger.error(
                            f"Error processing signal column",
                            exception=e,
                            context={"file": csv_file.name, "column": column_name}
                        )

            except Exception as e:
                self.logger.error(
                    f"Error processing CSV file",
                    exception=e,
                    context={"file": csv_file.name}
                )

        # Convert to DataFrame
        df_output = pd.DataFrame(all_results)

        # Define exact column order from notebook
        column_order = [
            'file_name', 'signal', 'signallength',
            'pe', 'comp', 'fisher_shannon', 'fisher_info',
            'renyipe', 'renyicomp', 'tsallispe', 'tsalliscomp',
            'dimension', 'tau', 'state', 'binaryclass', 'nan_percentage'
        ]

        # Ensure all columns exist and reorder
        for col in column_order:
            if col not in df_output.columns:
                df_output[col] = np.nan

        df_output = df_output[column_order]

        self.logger.info(
            f"Completed processing {signal_type}",
            {"csv_files": len(csv_files), "total_rows": len(df_output)}
        )

        return df_output

    def process_dataset(self,
                       dataset: str,
                       signal_types: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Process specified signal types in a dataset with granular file output.

        Saves separate CSV files for each dataset × signal_type combination:
        - results/results_train_bvp.csv
        - results/results_train_eda.csv
        - etc.

        Args:
            dataset: 'train', 'validation', or 'test'
            signal_types: List of signal types to process (None = all)

        Returns:
            Dictionary mapping signal type to feature DataFrame
        """
        self.logger.info(f"Starting {dataset} dataset processing")
        start_time = time.time()

        # Use provided signal types or default to all
        if signal_types is None:
            signal_types = self.settings.signal_types

        results = {}

        # Process each signal type
        for signal_type in signal_types:
            self.logger.info(f"Processing {dataset}/{signal_type}")

            df = self.process_signal_type(dataset, signal_type)
            results[signal_type] = df

            # Save granular output file: results_<dataset>_<signal_type>.csv
            if not df.empty:
                output_filename = f"results_{dataset}_{signal_type.lower()}.csv"
                output_file = Path(self.settings.results_dir) / output_filename
                df.to_csv(output_file, index=False)
                self.logger.info(f"Saved {output_file}")

        elapsed_time = time.time() - start_time
        total_rows = sum(len(df) for df in results.values())

        self.logger.info(
            f"Completed {dataset} dataset processing",
            {
                "elapsed_seconds": round(elapsed_time, 2),
                "total_rows": total_rows,
                "signal_types": list(results.keys())
            }
        )

        return results

    def run_pipeline(self,
                    datasets: Optional[List[str]] = None,
                    signal_types: Optional[List[str]] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Run feature extraction pipeline for specified datasets and signal types.

        Generates granular output files: results_<dataset>_<signal_type>.csv

        Args:
            datasets: List of datasets to process (None = all from settings)
            signal_types: List of signal types to process (None = all from settings)

        Returns:
            Nested dictionary: {dataset: {signal_type: DataFrame}}

        Examples:
            # Process all datasets, all signal types
            run_pipeline()

            # Process specific datasets
            run_pipeline(datasets=['train', 'test'])

            # Process specific signal types
            run_pipeline(signal_types=['Bvp', 'Eda'])

            # Process specific combination
            run_pipeline(datasets=['train'], signal_types=['Bvp'])
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting feature extraction pipeline")
        self.logger.info("=" * 60)

        pipeline_start = time.time()

        # Use provided datasets or default to settings
        if datasets is None:
            datasets = self.settings.datasets

        all_results = {}

        # Process each dataset
        for dataset_name in datasets:
            self.logger.info(f"Processing {dataset_name} dataset...")

            dataset_results = self.process_dataset(dataset_name, signal_types)
            all_results[dataset_name] = dataset_results

            # Force garbage collection between datasets
            gc.collect()

        # Pipeline summary
        pipeline_time = time.time() - pipeline_start

        summary = self._generate_summary(all_results, pipeline_time)
        self.logger.info("Pipeline Summary", summary)

        # Print summary to console
        self._print_summary(summary)

        return all_results

    def _generate_summary(self,
                         results: Dict[str, Dict[str, pd.DataFrame]],
                         elapsed_time: float) -> Dict[str, Any]:
        """
        Generate summary statistics for the pipeline run.

        Args:
            results: Pipeline results (nested dict: dataset -> signal_type -> DataFrame)
            elapsed_time: Total elapsed time in seconds

        Returns:
            Summary dictionary
        """
        summary = {
            'total_time_seconds': round(elapsed_time, 2),
            'total_time_minutes': round(elapsed_time / 60, 2),
            'datasets_processed': list(results.keys()),
            'row_counts': {}
        }

        for dataset, dataset_results in results.items():
            for signal_type, df in dataset_results.items():
                key = f"{dataset}/{signal_type}"
                summary['row_counts'][key] = len(df)

        summary['total_rows'] = sum(summary['row_counts'].values())

        return summary

    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """
        Print formatted summary to console.

        Args:
            summary: Summary dictionary
        """
        print("\n" + "=" * 60)
        print("FEATURE EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Total time: {summary['total_time_minutes']:.2f} minutes")
        print(f"Total rows generated: {summary['total_rows']:,}")
        print("\nRow breakdown by dataset/signal_type:")
        for key, count in summary['row_counts'].items():
            print(f"  {key}: {count:,} rows")
        print("=" * 60)