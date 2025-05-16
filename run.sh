#!/usr/bin/env bash

# Simple script to run the feature extraction 

# Activate the virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the extraction
python run_extraction.py "$@"

# Deactivate if we activated it
if [ -d "venv" ]; then
    deactivate
fi
