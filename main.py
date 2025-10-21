#!/usr/bin/env python3
"""
AI4PAIN Feature Extraction V2 - Professional Entropy-Based Analysis
====================================================================

Advanced feature extraction system using information-theoretic measures
for physiological pain assessment.

Setup:
------
1. Create a virtual environment: python -m venv venv
2. Activate it: source venv/bin/activate
3. Install dependencies: pip install -r requirements.txt
4. Add your data to data/{train,validation,test}/{Bvp,Eda,Resp,SpO2}/
5. Run: python main.py

Usage:
------
    python main.py                                  # Run with default settings
    python main.py --dimensions 3,4,5,6,7           # Custom dimensions
    python main.py --taus 1,2,3                     # Custom time delays
    python main.py --dataset train test             # Process specific datasets
    python main.py --signal-type Bvp Eda            # Process specific signal types
    python main.py --dataset train --signal-type Bvp  # Specific combination

Author: Vignan Kamarthi
Organization: Northeastern University
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.core.feature_extractor import FeatureExtractor
from src.config.settings import Settings
from src.utils.logger import SystemLogger


def parse_args():
    """
    Parse command line arguments for the feature extraction pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments containing:
        - dataset : List[str]
            Datasets to process (space-separated: train, validation, test)
        - signal_type : List[str]
            Signal types to process (space-separated: Bvp, Eda, Resp, SpO2)
        - dimensions : str
            Comma-separated embedding dimensions
        - taus : str
            Comma-separated time delays
        - filter : bool
            Whether to apply bandpass filtering
        - no_zscore : bool
            Whether to disable z-score normalization
        - no_progress : bool
            Whether to disable progress bars
    """
    parser = argparse.ArgumentParser(
        description='AI4Pain Feature Extraction V2 - Notebook-Aligned Implementation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                              # Process all datasets, all signal types
    python main.py --dataset train              # Process only training data
    python main.py --dataset train test         # Process training and test data
    python main.py --signal-type Bvp Eda        # Process only BVP and EDA signals
    python main.py --dimensions 4,5,6           # Custom dimensions
    python main.py --filter                     # Apply signal filtering
        """
    )

    # Data selection with multi-select
    parser.add_argument(
        '--dataset',
        type=str,
        nargs='+',
        choices=['train', 'validation', 'test'],
        default=['train', 'validation', 'test'],
        help='Datasets to process (space-separated, default: all)'
    )

    parser.add_argument(
        '--signal-type',
        type=str,
        nargs='+',
        choices=['Bvp', 'Eda', 'Resp', 'SpO2'],
        default=['Bvp', 'Eda', 'Resp', 'SpO2'],
        help='Signal types to process (space-separated, default: all)'
    )

    # Entropy parameters
    parser.add_argument(
        '--dimensions',
        type=str,
        default='3,4,5,6,7',
        help='Comma-separated embedding dimensions (default: 3,4,5,6,7)'
    )

    parser.add_argument(
        '--taus',
        type=str,
        default='1,2,3',
        help='Comma-separated time delays (default: 1,2,3)'
    )

    # Preprocessing options
    parser.add_argument(
        '--filter',
        action='store_true',
        help='Apply bandpass filtering (in addition to z-score)'
    )

    parser.add_argument(
        '--no-zscore',
        action='store_true',
        help='Disable z-score normalization'
    )

    # Output options
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars'
    )

    return parser.parse_args()


def validate_data_structure():
    """
    Validate that required data directories exist and contain CSV files.

    Checks for the presence of all required directories and ensures they
    contain at least one CSV file for processing.

    Returns
    -------
    bool
        True if data structure is valid and contains files, False otherwise.

    Notes
    -----
    Expected directory structure:
        data/
            train/[Bvp|Eda|Resp|SpO2]/*.csv
            validation/[Bvp|Eda|Resp|SpO2]/*.csv
            test/[Bvp|Eda|Resp|SpO2]/*.csv
    """
    logger = SystemLogger()
    data_path = Path("data")

    if not data_path.exists():
        logger.error("Data directory not found! Please create 'data/' folder")
        return False

    required_dirs = [
        "train/Bvp", "train/Eda", "train/Resp", "train/SpO2",
        "validation/Bvp", "validation/Eda", "validation/Resp", "validation/SpO2",
        "test/Bvp", "test/Eda", "test/Resp", "test/SpO2"
    ]

    missing = []
    empty = []

    for dir_path in required_dirs:
        full_path = data_path / dir_path
        if not full_path.exists():
            missing.append(str(dir_path))
        elif len(list(full_path.glob("*.csv"))) == 0:
            empty.append(str(dir_path))

    if missing:
        logger.warning(f"Missing directories: {missing}")

    if empty:
        logger.warning(f"Empty directories (no CSV files): {empty}")
        print("\n⚠️  Please add your CSV data files to the directories listed above!")
        return False

    return True


def main():
    """
    Main execution function for the AI4Pain feature extraction pipeline.

    This function orchestrates the entire feature extraction workflow:
    1. Parses command line arguments (multi-select datasets and signal types)
    2. Validates data directory structure
    3. Configures extraction settings
    4. Runs feature extraction pipeline
    5. Saves granular results to CSV files

    Raises
    ------
    SystemExit
        Exits with code 1 if data validation fails or if a fatal error occurs.

    KeyboardInterrupt
        Gracefully handles user interruption (Ctrl+C).

    Notes
    -----
    Results are saved to the 'results/' directory with granular filenames:
    - results_train_bvp.csv (training BVP signals)
    - results_validation_eda.csv (validation EDA signals)
    - etc. (one file per dataset × signal_type combination)
    """
    # Parse arguments
    args = parse_args()

    # Initialize logger
    logger = SystemLogger()
    logger.info("=" * 60)
    logger.info("AI4Pain Feature Extraction V2 - Starting")
    logger.info("=" * 60)

    # Validate data structure
    if not validate_data_structure():
        print("\nData validation failed! Please add your data files.")
        print("Expected structure:")
        print("  data/")
        print("    train/[Bvp|Eda|Resp|SpO2]/*.csv")
        print("    validation/[Bvp|Eda|Resp|SpO2]/*.csv")
        print("    test/[Bvp|Eda|Resp|SpO2]/*.csv")
        sys.exit(1)

    # Parse parameters
    dimensions = [int(d.strip()) for d in args.dimensions.split(',')]
    taus = [int(t.strip()) for t in args.taus.split(',')]

    # Create settings
    settings = Settings(
        dimensions=dimensions,
        taus=taus,
        apply_z_score=not args.no_zscore,
        apply_filtering=args.filter,
        show_progress=not args.no_progress
    )

    # Log configuration
    logger.info("Configuration", {
        "datasets": args.dataset,
        "signal_types": args.signal_type,
        "dimensions": dimensions,
        "taus": taus,
        "z_score": not args.no_zscore,
        "filtering": args.filter
    })

    # Create feature extractor
    extractor = FeatureExtractor(settings)

    try:
        # Run extraction with multi-select parameters
        results = extractor.run_pipeline(
            datasets=args.dataset,
            signal_types=args.signal_type
        )

        print("\nFeature extraction completed successfully!")
        print(f"Results saved to: {settings.results_dir}/")
        print(f"\nProcessed:")
        print(f"  Datasets: {', '.join(args.dataset)}")
        print(f"  Signal types: {', '.join(args.signal_type)}")

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        print("\nProcess interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error("Fatal error during extraction", exception=e)
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
