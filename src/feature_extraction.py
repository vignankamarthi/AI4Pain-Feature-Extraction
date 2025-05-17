import pandas as pd
import numpy as np
import os
import logging
import traceback
import gc
import warnings
from ordpy import complexity_entropy, fisher_shannon
from datetime import datetime
from tqdm import tqdm

# TODO Create terminal commands in main.py to also choose the dimensions and taus
# TODO Create a "view_results.py file to visualize the results as a Pandas DataFrame"
# TODO Understand the data points we are using
# TODO Add valuable statiscal parameters and information to the view_results.py file
# TODO Finalize README
# TODO Finalize GitHub repository
# TODO Send full report to Sri


# Set up logging
def setup_logging():
    """
    Configure logging to write to the feature_extraction file.
    The log file is reset at each run.

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Configure logging
    log_file = "logs/feature_extraction.log"

    # Reset log file
    with open(log_file, "w") as f:
        f.write(f"=== Log started at {datetime.now()} ===\n\n")

    # Configure logger
    logger = logging.getLogger("feature_extraction")
    logger.setLevel(logging.INFO)

    # Create file handler only
    file_handler = logging.FileHandler(log_file, mode="a")

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    return logger


# Initialize the Global logger
logger = None


def process_single_dimension_tau(signal_data, dim, tau):
    """
    Calculate metrics for a single dimension-tau combination and replace invalid values with NaN.

    Parameters
    ----------
    signal_data : array_like
        The signal data to process.
    dim : int
        Embedding dimension.
    tau : int
        Time delay.

    Returns
    -------
    dict
        Dictionary containing calculated metrics for the dimension-tau combination.
    """
    # Set default result with NaN values
    result = {
        "signallength": len(signal_data),
        "pe": np.nan,
        "pe_fisher": np.nan,
        "comp": np.nan,
        "fisher": np.nan,
        "dimension": dim,
        "tau": tau,
    }

    try:
        # Check if signal has enough unique values for this dimension
        unique_values = np.unique(signal_data[~np.isnan(signal_data)])
        if len(unique_values) < dim:
            logger.warning(
                f"Signal doesn't have enough unique values for dim={dim}, tau={tau}"
            )
            return result

        # Capture warnings to log them properly
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Calculate permutation entropy and complexity
            pe_comp = complexity_entropy(signal_data, dim, tau)

            # Get values with validation
            pe_value = pe_comp[0]  # Permutation entropy
            comp_value = pe_comp[1]  # Complexity

            # Log any warnings
            for warning in w:
                logger.warning(
                    f"Warning in PE calculation (dim={dim}, tau={tau}): {warning.message}"
                )

        # Reset warnings capture for Fisher calculation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Calculate Fisher complexity
            fisher_result = fisher_shannon(signal_data, dim, tau)

            # Get values
            pe_fisher_value = fisher_result[0]  # PE from fisher_shannon
            fisher_value = fisher_result[1]  # Fisher complexity

            # Log any warnings
            for warning in w:
                logger.warning(
                    f"Warning in Fisher calculation (dim={dim}, tau={tau}): {warning.message}"
                )

        # Check for invalid values and replace with NaN
        if np.isnan(pe_value) or np.isinf(pe_value):
            logger.warning(f"Invalid PE value: {pe_value} for dim={dim}, tau={tau}")
            pe_value = np.nan

        if np.isnan(comp_value) or np.isinf(comp_value):
            logger.warning(
                f"Invalid complexity value: {comp_value} for dim={dim}, tau={tau}"
            )
            comp_value = np.nan

        if np.isnan(pe_fisher_value) or np.isinf(pe_fisher_value):
            logger.warning(
                f"Invalid PE (Fisher) value: {pe_fisher_value} for dim={dim}, tau={tau}"
            )
            pe_fisher_value = np.nan

        if np.isnan(fisher_value) or np.isinf(fisher_value):
            logger.warning(
                f"Invalid Fisher value: {fisher_value} for dim={dim}, tau={tau}"
            )
            fisher_value = np.nan

        # Compare PE values if both are valid
        if not np.isnan(pe_value) and not np.isnan(pe_fisher_value):
            pe_diff = abs(pe_value - pe_fisher_value)
            if pe_diff > 1e-10:
                logger.warning(
                    f"PE mismatch: {pe_value} vs {pe_fisher_value} (diff: {pe_diff}) for dim={dim}, tau={tau}"
                )

        # Store validated results
        result = {
            "signallength": len(signal_data),
            "pe": pe_value,
            "pe_fisher": pe_fisher_value,
            "comp": comp_value,
            "fisher": fisher_value,
            "dimension": dim,
            "tau": tau,
        }

        return result

    except MemoryError as e:
        logger.error(
            f"Memory error processing signal with dim={dim}, tau={tau}: {str(e)}"
        )
        return result
    except Exception as e:
        logger.error(f"Error processing signal with dim={dim}, tau={tau}: {str(e)}")
        logger.error(traceback.format_exc())
        return result


def process_signal(signal_data, dimensions=[1, 2], taus=[1, 2, 3]):
    """
    Process a single signal with multiple dimension and tau combinations,
    one at a time to manage memory, with lower dimensions processed first.

    Parameters
    ----------
    signal_data : array_like
        The signal data to process.
    dimensions : list, optional
        List of embedding dimensions to use for calculations, by default [1, 2].
    taus : list, optional
        List of time delays to use for calculations, by default [1, 2, 3].

    Returns
    -------
    list of dict
        List of dictionaries containing calculated metrics for each dimension-tau combination.
        Each dictionary contains keys: 'signallength', 'pe', 'pe_fisher', 'comp', 'fisher',
        'dimension', and 'tau'.
    """
    results = []

    # Sort dimensions to process lower dimensions first (less memory intensive)
    sorted_dimensions = sorted(dimensions)

    for dim in sorted_dimensions:
        for tau in taus:
            try:
                # Process this specific dimension-tau combination
                logger.info(f"Processing dimension={dim}, tau={tau}")
                result = process_single_dimension_tau(signal_data, dim, tau)
                results.append(result)

                # Force garbage collection after each computation
                gc.collect()

            except Exception as e:
                logger.error(
                    f"Unexpected error in process_signal for dim={dim}, tau={tau}: {str(e)}"
                )
                logger.error(traceback.format_exc())

                # Create a result with error indicators
                result = {
                    "signallength": len(signal_data),
                    "pe": np.nan,
                    "pe_fisher": np.nan,
                    "comp": np.nan,
                    "fisher": np.nan,
                    "dimension": dim,
                    "tau": tau,
                }
                results.append(result)

                # Force garbage collection
                gc.collect()

    return results


def count_total_files(data_dir):
    """Count the total number of CSV files in the data directory for progress tracking."""
    total_files = 0
    if not os.path.exists(data_dir):
        return total_files

    for root, dirs, files in os.walk(data_dir):
        total_files += sum(1 for f in files if f.endswith(".csv"))

    return total_files


def process_files(data_dir, dimensions=[1, 2], taus=[1, 2, 3]):
    """
    Process all CSV files in the given directory and its subdirectories.
    The directory structure is expected to be:
    data_dir/
        Bvp/
            *.csv
        Eda/
            *.csv
        Resp/
            *.csv
        SpO2/
            *.csv

    Parameters
    ----------
    data_dir : str
        Path to the directory containing subdirectories with CSV files to process.
    dimensions : list, optional
        List of embedding dimensions to use for calculations, by default [1, 2].
    taus : list, optional
        List of time delays to use for calculations, by default [1, 2, 3].

    Returns
    -------
    list of dict
        List of dictionaries containing calculated features for all signals
        in all files. PE verification is always included.
    """
    all_results = []

    try:
        # Check if directory exists
        if not os.path.exists(data_dir):
            logger.error(f"Directory not found: {data_dir}")
            print(f"ERROR: Directory not found: {data_dir}")
            return all_results

        # Get subdirectories (Bvp, Eda, Resp, SpO2)
        try:
            subdirs = [
                d
                for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ]
            logger.info(f"Found {len(subdirs)} subdirectories in {data_dir}: {subdirs}")
            print(f"Found {len(subdirs)} subdirectories in {data_dir}: {subdirs}")
        except Exception as e:
            logger.error(f"Error listing directory {data_dir}: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"ERROR: Could not list subdirectories in {data_dir}")
            return all_results

        # Count total files for progress tracking
        total_files = count_total_files(data_dir)
        print(f"Found {total_files} total CSV files to process")

        # Set up progress bar for overall file processing
        overall_progress = tqdm(total=total_files, desc="Overall Progress", unit="file")

        # Process each subdirectory
        total_csv_files = 0
        for subdir in subdirs:
            subdir_path = os.path.join(data_dir, subdir)

            # Get all CSV files in this subdirectory
            try:
                csv_files = [f for f in os.listdir(subdir_path) if f.endswith(".csv")]
                logger.info(f"Found {len(csv_files)} CSV files in {subdir_path}")
                print(f"Found {len(csv_files)} CSV files in {subdir_path}")
                total_csv_files += len(csv_files)
            except Exception as e:
                logger.error(f"Error listing directory {subdir_path}: {str(e)}")
                logger.error(traceback.format_exc())
                print(f"ERROR: Could not list files in {subdir_path}")
                continue

            # Create a progress bar for this subdirectory
            subdir_progress = tqdm(
                csv_files, desc=f"Processing {subdir}", unit="file", leave=False
            )

            # Process each file in this subdirectory
            for file_name in subdir_progress:
                file_path = os.path.join(subdir_path, file_name)
                logger.info(f"Processing file: {subdir}/{file_name}")

                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    logger.info(f"Successfully read {file_name}")

                    # Process each signal column
                    signal_count = 0
                    signal_columns = [
                        col
                        for col in df.columns
                        if col.startswith(file_name.split(".")[0])
                    ]

                    # Create a progress bar for processing signals within the file
                    signal_progress = tqdm(
                        signal_columns,
                        desc=f"Signals in {file_name}",
                        unit="signal",
                        leave=False,
                    )

                    for column in signal_progress:
                        signal_count += 1

                        logger.info(f"Processing signal: {column}")

                        try:
                            # Extract signal data
                            signal_data = df[column].values

                            # Extract state from signal name (text between underscores)
                            try:
                                parts = column.split("_")
                                if len(parts) >= 2:
                                    state = parts[
                                        1
                                    ].lower()  # Convert state to lowercase
                                else:
                                    state = "unknown"
                                    logger.warning(
                                        f"Could not extract state from signal name: {column}"
                                    )
                            except Exception as e:
                                state = "unknown"
                                logger.error(
                                    f"Error extracting state from signal {column}: {str(e)}"
                                )

                            # Process the signal
                            try:
                                results = process_signal(signal_data, dimensions=dimensions, taus=taus)
                                logger.info(f"Successfully processed signal {column}")
                            except Exception as e:
                                logger.error(
                                    f"Error processing signal {column}: {str(e)}"
                                )
                                logger.error(traceback.format_exc())
                                continue

                            # Add file, signal type and signal information to each result
                            for result in results:
                                result["file_name"] = file_name
                                result["signal"] = column
                                result["signal_type"] = subdir
                                result["state"] = state

                            # Add to overall results list
                            all_results.extend(results)

                            # Force garbage collection after each signal
                            gc.collect()

                        except Exception as e:
                            logger.error(
                                f"Error processing column {column} in file {file_name}: {str(e)}"
                            )
                            logger.error(traceback.format_exc())
                            # Continue to the next signal

                    logger.info(
                        f"Processed {signal_count} signals in {subdir}/{file_name}"
                    )

                    # Clear dataframe from memory
                    del df
                    gc.collect()

                except Exception as e:
                    logger.error(f"Error processing file {file_name}: {str(e)}")
                    logger.error(traceback.format_exc())
                    print(f"ERROR: Failed to process file {file_name}")

                # Update the overall progress bar
                overall_progress.update(1)

            # Close subdir progress bar
            subdir_progress.close()

        # Close overall progress bar
        overall_progress.close()

    except Exception as e:
        logger.error(f"Unexpected error in process_files: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"ERROR: Unexpected error in processing files")

    logger.info(
        f"Total processed results: {len(all_results)} from {total_csv_files} files"
    )
    print(
        f"Completed processing with {len(all_results)} total results from {total_csv_files} files"
    )
    return all_results


def generate_feature_table(all_results, output_file, include_pe_verification=False):
    """
    Generate a feature table from processed results and save to CSV.

    Parameters
    ----------
    all_results : list of dict
        List of dictionaries containing calculated features.
    output_file : str
        Path to save the output CSV file.
    include_pe_verification : bool, optional
        Whether to include PE values from fisher_shannon for verification,
        by default False.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing the generated feature table, or None if an error occurred.
    """
    try:
        # Check if results are empty
        if not all_results:
            logger.error("No results to generate table from")
            print("ERROR: No results to generate table from")
            return None

        # Convert results to DataFrame
        try:
            print("Creating feature table from results...")
            result_df = pd.DataFrame(all_results)
            logger.info(f"Created DataFrame with {len(result_df)} rows")
            print(f"Created feature table with {len(result_df)} rows")
        except Exception as e:
            logger.error(f"Error creating DataFrame from results: {str(e)}")
            logger.error(traceback.format_exc())
            print("ERROR: Failed to create feature table")
            return None

        # Reorder and filter columns
        try:
            print("Organizing columns...")
            if include_pe_verification:
                ordered_columns = [
                    "file_name",
                    "signal",
                    "signal_type",
                    "signallength",
                    "pe",
                    "pe_fisher",
                    "comp",
                    "fisher",
                    "dimension",
                    "tau",
                    "state",
                ]
                logger.info("Including PE verification column in output")
                print("Including PE verification column in output")
            else:
                # Remove verification column if not needed
                result_df = result_df.drop(columns=["pe_fisher"])
                ordered_columns = [
                    "file_name",
                    "signal",
                    "signal_type",
                    "signallength",
                    "pe",
                    "comp",
                    "fisher",
                    "dimension",
                    "tau",
                    "state",
                ]
                logger.info("Excluded PE verification column from output")
                print("Excluded PE verification column from output")

            result_df = result_df[ordered_columns]
        except Exception as e:
            logger.error(f"Error preparing columns: {str(e)}")
            logger.error(traceback.format_exc())
            print("ERROR: Failed to prepare columns for output")
            return None

        # Create directory if it doesn't exist
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        except Exception as e:
            logger.error(
                f"Error creating directory for output file {output_file}: {str(e)}"
            )
            logger.error(traceback.format_exc())
            print(f"ERROR: Failed to create directory for {output_file}")

        # Save to CSV
        try:
            print(f"Saving feature table to {output_file}...")
            result_df.to_csv(output_file, index=False)
            logger.info(f"Successfully saved feature table to {output_file}")
            print(f"Successfully saved feature table to {output_file}")
        except Exception as e:
            logger.error(f"Error saving DataFrame to CSV file {output_file}: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"ERROR: Failed to save table to {output_file}")
            return None

        return result_df

    except Exception as e:
        logger.error(f"Unexpected error in generate_feature_table: {str(e)}")
        logger.error(traceback.format_exc())
        print("ERROR: Unexpected error generating feature table")
        return None


def main(include_pe_verification=False, dimensions=[1, 2], taus=[1, 2, 3]):
    """
    Main function to process data and generate feature tables.

    Parameters
    ----------
    include_pe_verification : bool, optional
        Whether to include PE values from fisher_shannon for verification
        in the output tables, by default False.
    dimensions : list, optional
        List of embedding dimensions to use for calculations, by default [1, 2].
    taus : list, optional
        List of time delays to use for calculations, by default [1, 2, 3].
    """
    global logger
    # Set up logging
    logger = setup_logging()

    try:
        print("Starting feature extraction process")
        logger.info("Starting feature extraction process")

        # Set paths
        train_dir = "data/train"  # Contains subfolders for Bvp, Eda, Resp, SpO2
        test_dir = "data/test"  # Contains subfolders for Bvp, Eda, Resp, SpO2
        output_dir = "results"

        # Create output directory if it doesn't exist
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Error creating output directory {output_dir}: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"ERROR: Could not create output directory {output_dir}")
            return

        # Setup progress tracking for entire process
        main_progress = tqdm(total=4, desc="Overall Process", unit="phase")

        # Process training data
        print("\nProcessing training data...")
        logger.info("Processing training data...")
        train_results = process_files(train_dir, dimensions=dimensions, taus=taus)
        main_progress.update(
            1
        )  # Update main progress bar after training data processing

        # Force garbage collection after processing training data
        gc.collect()

        train_df = generate_feature_table(
            train_results,
            os.path.join(output_dir, "features_train.csv"),
            include_pe_verification,
        )
        main_progress.update(
            1
        )  # Update main progress bar after generating training features

        # Check if training data processing was successful
        if train_df is None:
            logger.error("Failed to process training data")
            print("ERROR: Failed to process training data")
            main_progress.close()
            return

        # Clear train_results to free memory
        del train_results
        gc.collect()

        # Process test data
        print("\nProcessing test data...")
        logger.info("Processing test data...")
        test_results = process_files(test_dir, dimensions=dimensions, taus=taus)
        main_progress.update(1)  # Update main progress bar after test data processing

        # Force garbage collection after processing test data
        gc.collect()

        test_df = generate_feature_table(
            test_results,
            os.path.join(output_dir, "features_test.csv"),
            include_pe_verification,
        )

        # Check if test data processing was successful
        if test_df is None:
            logger.error("Failed to process test data")
            print("ERROR: Failed to process test data")
            main_progress.close()
            return

        # Clear test_results to free memory
        del test_results
        gc.collect()

        # Combine results
        print("\nCombining results...")
        try:
            all_results = pd.concat([train_df, test_df])
            logger.info(
                f"Combined train and test data into a single DataFrame with {len(all_results)} rows"
            )
        except Exception as e:
            logger.error(f"Error combining train and test results: {str(e)}")
            logger.error(traceback.format_exc())
            print("ERROR: Failed to combine train and test results")
            main_progress.close()
            return

        # Save combined results
        try:
            print("Saving combined feature table...")
            all_results.to_csv(
                os.path.join(output_dir, "features_complete.csv"), index=False
            )
            logger.info(f"Saved combined results to features_complete.csv")
            print(
                f"Saved combined results to {os.path.join(output_dir, 'features_complete.csv')}"
            )
        except Exception as e:
            logger.error(f"Error saving combined results: {str(e)}")
            logger.error(traceback.format_exc())
            print("ERROR: Failed to save combined results")
            main_progress.close()
            return

        main_progress.update(1)  # Final update to main progress bar
        main_progress.close()

        print(f"\nProcessing complete. Total features: {len(all_results)}")
        logger.info(f"Processing complete. Total features: {len(all_results)}")

        # If verification was kept, check PE values
        try:
            if "pe_fisher" in all_results.columns:
                # Use only finite values for comparison
                valid_mask = np.isfinite(all_results["pe"]) & np.isfinite(
                    all_results["pe_fisher"]
                )
                if valid_mask.sum() > 0:
                    pe_diff = (
                        all_results.loc[valid_mask, "pe"]
                        - all_results.loc[valid_mask, "pe_fisher"]
                    ).abs()
                    max_diff = pe_diff.max()
                    logger.info(f"Maximum PE difference: {max_diff}")
                    print(f"Maximum PE difference: {max_diff}")

                    different_count = (pe_diff > 1e-10).sum()
                    if different_count > 0:
                        logger.warning(
                            f"Found {different_count} instances where PE values differ"
                        )
                        print(
                            f"WARNING: Found {different_count} instances where PE values differ"
                        )
                    else:
                        logger.info("All PE values match between calculation methods")
                        print("All PE values match between calculation methods")

                # Log counts of invalid values
                nan_pe_count = all_results["pe"].isna().sum()
                nan_pe_fisher_count = all_results["pe_fisher"].isna().sum()
                logger.info(
                    f"NaN values in PE: {nan_pe_count}, NaN values in PE Fisher: {nan_pe_fisher_count}"
                )
                print(
                    f"NaN values in PE: {nan_pe_count}, NaN values in PE Fisher: {nan_pe_fisher_count}"
                )
        except Exception as e:
            logger.error(f"Error performing PE verification check: {str(e)}")
            logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"Unexpected error in main function: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"ERROR: Unexpected error in main function")

    # Close all handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    # Set to True to include verification columns in output
    main(include_pe_verification=False)
