#!/usr/bin/env bash

# This is a simple "make file" to make common tasks easy

case "$1" in
  setup)
    # Set up the project (create virtual environment and install dependencies)
    python setup.py
    ;;
  run)
    # Run the feature extraction
    ./run.sh "${@:2}"
    ;;
  clean)
    # Clean up generated files but keep the directory structure
    rm -rf results/*
    rm -rf logs/*
    mkdir -p results logs
    echo "Cleaned results and logs directories"
    ;;
  reset)
    # Complete reset - remove virtual environment and all generated files
    rm -rf venv results/* logs/*
    mkdir -p results logs
    echo "Reset complete. Run './ai4pain.sh setup' to set up again."
    ;;
  *)
    # Show help
    echo "AI4PAIN Feature Extraction Utility"
    echo "Usage: ./ai4pain.sh [command]"
    echo ""
    echo "Commands:"
    echo "  setup   - Create virtual environment and install dependencies"
    echo "  run     - Run feature extraction (pass any arguments to run_extraction.py)"
    echo "  clean   - Clean results and logs directories"
    echo "  reset   - Remove virtual environment and all generated files"
    ;;
esac
