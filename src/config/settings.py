"""
Settings: Configuration management for AI4Pain Feature Extraction.

This module centralizes all configuration parameters for the feature extraction pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class Settings:
    """
    Central configuration for the feature extraction pipeline.

    All parameters can be overridden via initialization or modified at runtime.
    """

    # Data paths
    data_dir: str = "data"
    results_dir: str = "results"
    logs_dir: str = "logs"

    # Signal types to process
    signal_types: List[str] = field(default_factory=lambda: ['Bvp', 'Eda', 'Resp', 'SpO2'])

    # Entropy calculation parameters (matching notebook)
    dimensions: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7])
    taus: List[int] = field(default_factory=lambda: [1, 2, 3])

    # Preprocessing options
    apply_z_score: bool = True
    apply_filtering: bool = False  # Set to True to enable bandpass filtering

    # Sampling frequency (Hz)
    sampling_frequency: float = 100.0

    # Output options
    include_pe_verification: bool = False  # Include PE from fisher_shannon for verification
    save_individual_signals: bool = True   # Save features for each signal type separately
    save_concatenated: bool = True          # Save concatenated dataset for ML

    # Performance options
    use_multiprocessing: bool = False  # Enable for parallel processing (future enhancement)
    max_workers: int = 4               # Number of parallel workers

    # Validation thresholds
    max_nan_percentage: float = 50.0   # Maximum allowed NaN percentage in signals
    min_unique_values: int = 3         # Minimum unique values for valid signal

    # File naming patterns
    train_prefix: str = "features_train"
    validation_prefix: str = "features_validation"
    test_prefix: str = "features_test"
    combined_filename: str = "features_complete.csv"

    # Datasets to process
    datasets: List[str] = field(default_factory=lambda: ['train', 'validation', 'test'])

    # Progress bar options
    show_progress: bool = True
    nested_progress: bool = True  # Show nested progress bars

    # Logging options
    log_level: str = "INFO"
    clear_log_on_start: bool = False

    def __post_init__(self):
        """Validate and process settings after initialization."""
        # Convert string paths to Path objects
        self.data_path = Path(self.data_dir)
        self.results_path = Path(self.results_dir)
        self.logs_path = Path(self.logs_dir)

        # Validate dimensions and taus
        if not self.dimensions:
            raise ValueError("At least one dimension must be specified")
        if not self.taus:
            raise ValueError("At least one tau value must be specified")

        # Sort dimensions for efficient processing (lower dimensions first)
        self.dimensions = sorted(self.dimensions)

    def get_output_filename(self,
                           dataset: str,
                           signal_type: Optional[str] = None) -> str:
        """
        Generate output filename based on dataset and signal type.

        Args:
            dataset: 'train', 'validation', or 'test'
            signal_type: Optional signal type (e.g., 'Bvp', 'all')

        Returns:
            Output filename
        """
        prefix_map = {
            'train': self.train_prefix,
            'validation': self.validation_prefix,
            'test': self.test_prefix
        }
        prefix = prefix_map.get(dataset, f"features_{dataset}")

        if signal_type:
            return f"{prefix}_{signal_type.lower()}.csv"
        else:
            return f"{prefix}.csv"

    def get_feature_columns(self) -> List[str]:
        """
        Get the list of feature columns that will be generated.

        Returns:
            List of feature column names
        """
        base_columns = [
            'file_name', 'signal', 'signal_type', 'signallength',
            'pe', 'comp', 'fisher', 'renyipe', 'renyicomp',
            'tsallispe', 'tsalliscomp', 'dimension', 'tau',
            'state', 'binaryclass'
        ]

        if self.include_pe_verification:
            # Insert pe_fisher after pe
            pe_index = base_columns.index('pe')
            base_columns.insert(pe_index + 1, 'pe_fisher')

        return base_columns

    def to_dict(self) -> dict:
        """
        Convert settings to dictionary for logging.

        Returns:
            Dictionary representation of settings
        """
        return {
            'data_dir': self.data_dir,
            'results_dir': self.results_dir,
            'dimensions': self.dimensions,
            'taus': self.taus,
            'signal_types': self.signal_types,
            'apply_z_score': self.apply_z_score,
            'apply_filtering': self.apply_filtering,
            'sampling_frequency': self.sampling_frequency,
            'total_combinations': len(self.dimensions) * len(self.taus)
        }

    def validate_paths(self) -> bool:
        """
        Validate that required paths exist.

        Returns:
            True if all paths are valid
        """
        # Create directories if they don't exist
        self.results_path.mkdir(exist_ok=True)
        self.logs_path.mkdir(exist_ok=True)

        # Check data directory
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")

        return True

    def __str__(self) -> str:
        """String representation of settings."""
        return (
            f"Settings(\n"
            f"  Data: {self.data_dir}\n"
            f"  Dimensions: {self.dimensions}\n"
            f"  Taus: {self.taus}\n"
            f"  Signals: {self.signal_types}\n"
            f"  Preprocessing: z-score={self.apply_z_score}, filter={self.apply_filtering}\n"
            f")"
        )


# Default settings instance
default_settings = Settings()