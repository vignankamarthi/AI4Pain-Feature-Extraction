#!/usr/bin/env python3
"""
This is a convenience script to set up the project.
It creates a virtual environment and installs all required dependencies.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def main():
    """Create a virtual environment and install dependencies."""
    print("Setting up AI4PAIN Feature Extraction...")
    
    # Create directories if needed
    for path in [
        "data/train/Bvp", "data/train/Eda", "data/train/Resp", "data/train/SpO2",
        "data/test/Bvp", "data/test/Eda", "data/test/Resp", "data/test/SpO2"
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)
        
    # Remove existing virtual environment if it exists
    venv_path = Path("venv")
    if venv_path.exists():
        print("Removing existing virtual environment...")
        shutil.rmtree(venv_path)
    
    # Create virtual environment
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Determine activation script and pip command
    if sys.platform == "win32":
        pip_cmd = [str(venv_path / "Scripts" / "pip")]
        activate_script = str(venv_path / "Scripts" / "activate")
    else:
        pip_cmd = [str(venv_path / "bin" / "pip")]
        activate_script = str(venv_path / "bin" / "activate")
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.run(pip_cmd + ["install", "--upgrade", "pip"], check=True)
    subprocess.run(pip_cmd + ["install", "-r", "requirements.txt"], check=True)
    
    print("\nSetup complete!")
    print(f"To activate the virtual environment, use: source {activate_script}")
    print("To run the feature extraction: python run_extraction.py")
    print("Or simply use the convenience script: ./run.sh")

if __name__ == "__main__":
    main()
