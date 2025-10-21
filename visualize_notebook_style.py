#!/usr/bin/env python3
"""
Notebook-Style Visualization for AI4Pain Feature Extraction

Generates the exact visualization from the reference notebook:
- 5×3 grid (dimensions 3-7 × taus 1-3)
- Renyi Complexity vs. Renyi Permutation Entropy
- Color-coded by pain state (excluding 'rest')
- Combined across all signal types

Usage:
------
    python visualize_notebook_style.py --dataset train
    python visualize_notebook_style.py --dataset train validation --signal-type Bvp
    python visualize_notebook_style.py --help

Author: Vignan Kamarthi
Organization: Northeastern University
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from typing import List, Optional


def load_results(dataset: str,
                signal_types: Optional[List[str]] = None,
                results_dir: str = "results") -> pd.DataFrame:
    """
    Load and concatenate results for specified dataset and signal types.

    Args:
        dataset: Dataset name ('train', 'validation', or 'test')
        signal_types: List of signal types to load (None = all)
        results_dir: Directory containing result CSV files

    Returns:
        Combined DataFrame with all requested data

    Raises:
        FileNotFoundError: If result files are not found
    """
    if signal_types is None:
        signal_types = ['Bvp', 'Eda', 'Resp', 'SpO2']

    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    dataframes = []

    for signal_type in signal_types:
        filename = f"results_{dataset}_{signal_type.lower()}.csv"
        filepath = results_path / filename

        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue

        try:
            df = pd.read_csv(filepath)
            dataframes.append(df)
            print(f"Loaded {len(df)} rows from {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if not dataframes:
        raise FileNotFoundError(
            f"No result files found for dataset '{dataset}' with signal types {signal_types}"
        )

    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined_df)}")

    return combined_df


def create_complexity_entropy_plot(df: pd.DataFrame,
                                  dataset: str,
                                  output_dir: str = "results") -> str:
    """
    Create 5×3 grid of Renyi complexity-entropy plots.

    Exact replication of notebook cell 8:
    - 5 rows (dimensions 3-7)
    - 3 columns (taus 1-3)
    - Scatter plot: renyipe (x) vs. renyicomp (y)
    - Color by state (excluding 'rest')

    Args:
        df: DataFrame with long-format features
        dataset: Dataset name for plot title
        output_dir: Directory to save plot

    Returns:
        Path to saved plot file
    """
    # Filter out 'rest' state (as in notebook)
    df_filtered = df[df['state'] != 'rest'].copy()

    if len(df_filtered) == 0:
        raise ValueError("No data remaining after filtering out 'rest' state")

    print(f"Plotting {len(df_filtered)} rows (excluded 'rest' state)")
    print(f"States included: {df_filtered['state'].unique()}")

    # Get unique dimensions and taus (sorted)
    dimensions = sorted(df_filtered['dimension'].unique())
    taus = sorted(df_filtered['tau'].unique())

    # Create figure with 5×3 subplots
    fig, axes = plt.subplots(
        len(dimensions), len(taus),
        figsize=(15, 15),
        sharex=True, sharey=True
    )
    axes = axes.flatten()

    # Plot each dimension × tau combination
    plot_idx = 0
    for dim in dimensions:
        for tau in taus:
            # Filter data for this combination
            subset = df_filtered[
                (df_filtered['dimension'] == dim) &
                (df_filtered['tau'] == tau)
            ]

            # Create scatter plot
            ax = axes[plot_idx]
            sns.scatterplot(
                data=subset,
                x='renyipe',
                y='renyicomp',
                hue='state',
                ax=ax,
                alpha=0.6
            )

            # Set title and labels
            ax.set_title(f'Dimension: {dim}, Tau: {tau}', fontsize=12)
            ax.set_xlabel('Renyi Permutation Entropy', fontsize=10)
            ax.set_ylabel('Renyi Complexity', fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.legend(title='State', fontsize=8, loc='best')

            plot_idx += 1

    # Overall title
    fig.suptitle(
        f'Renyi Complexity-Entropy Plane: {dataset.capitalize()} Dataset',
        fontsize=16,
        fontweight='bold'
    )

    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    plot_filename = f"renyi_complexity_entropy_{dataset}.png"
    plot_filepath = output_path / plot_filename

    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_filepath}")

    plt.close()

    return str(plot_filepath)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate notebook-style Renyi complexity-entropy visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_notebook_style.py --dataset train
    python visualize_notebook_style.py --dataset validation --signal-type Bvp Eda
    python visualize_notebook_style.py --dataset train --output plots/
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['train', 'validation', 'test'],
        help='Dataset to visualize (required)'
    )

    parser.add_argument(
        '--signal-type',
        type=str,
        nargs='+',
        choices=['Bvp', 'Eda', 'Resp', 'SpO2'],
        default=['Bvp', 'Eda', 'Resp', 'SpO2'],
        help='Signal types to include (default: all)'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory containing result CSV files (default: results)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for plots (default: results)'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    print("=" * 60)
    print("AI4Pain Notebook-Style Visualization")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Signal types: {', '.join(args.signal_type)}")
    print()

    try:
        # Load results
        print("Loading results...")
        df = load_results(
            dataset=args.dataset,
            signal_types=args.signal_type,
            results_dir=args.results_dir
        )

        # Create plot
        print("\nGenerating complexity-entropy plot...")
        plot_path = create_complexity_entropy_plot(
            df=df,
            dataset=args.dataset,
            output_dir=args.output
        )

        print("\nVisualization complete!")
        print(f"Plot saved: {plot_path}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run feature extraction first:")
        print(f"  python main.py --dataset {args.dataset}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
