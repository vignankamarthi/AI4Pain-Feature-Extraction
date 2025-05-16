#!/usr/bin/env bash

# This script sets up the project by creating a virtual environment and installing dependencies

# Navigate to the project directory
cd "$(dirname "$0")"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source venv/bin/activate

# Install setuptools and wheel first
echo "Installing setuptools and wheel first..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create data directories if they don't exist
echo "Creating data directory structure..."

# Train data subdirectories
mkdir -p data/train/Bvp
mkdir -p data/train/Eda
mkdir -p data/train/Resp
mkdir -p data/train/SpO2

# Test data subdirectories
mkdir -p data/test/Bvp
mkdir -p data/test/Eda
mkdir -p data/test/Resp
mkdir -p data/test/SpO2

echo "Data directory structure created."
echo "Setup complete! You can now run the feature extraction using ./run.sh"
