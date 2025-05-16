#!/usr/bin/env bash

# This script is a convenience wrapper to activate the virtual environment
# and run the feature extraction

# Navigate to the project directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ -d "venv" ]; then
    # Activate virtual environment
    source venv/bin/activate
else
    echo "Virtual environment not found. Please run setup first."
    exit 1
fi

# Run the feature extraction script
python run_extraction.py "$@"

# Deactivate virtual environment
deactivate
