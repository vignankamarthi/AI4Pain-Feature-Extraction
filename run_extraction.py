#!/usr/bin/env python3
"""
AI4PAIN Feature Extraction Runner
--------------------------------
This script runs the feature extraction process for the AI4PAIN 2025 Challenge.

Expected directory structure:
  data/
    train/
      Bvp/      # Blood Volume Pulse signals
      Eda/      # Electrodermal Activity signals
      Resp/     # Respiration signals
      SpO2/     # Blood Oxygen Saturation signals
    test/
      Bvp/      # Blood Volume Pulse signals
      Eda/      # Electrodermal Activity signals
      Resp/     # Respiration signals
      SpO2/     # Blood Oxygen Saturation signals
"""

import argparse
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the feature extraction module
from feature_extraction import main as run_feature_extraction

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI4PAIN 2025 Feature Extraction')
    
    parser.add_argument('--verify', action='store_true', 
                        help='Include PE verification column in output')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_feature_extraction(include_pe_verification=args.verify)
