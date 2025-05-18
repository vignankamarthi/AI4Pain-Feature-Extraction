#!/usr/bin/env python3
"""
AI4PAIN Feature Extraction - Simple Entry Point
-----------------------------------------------
Run this script to perform feature extraction on physiological signals.

Manual setup:
1. Create a virtual environment: python -m venv venv
2. Activate it: source venv/bin/activate (or venv\\Scripts\\activate on Windows)
3. Install dependencies: pip install -r requirements.txt
4. Run this script: python main.py [--verify]
"""

import argparse
import os
from src.feature_extraction import main as run_feature_extraction

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI4PAIN Feature Extraction')
    parser.add_argument('--verify', action='store_true', 
                        help='Include Fisher PE verification column in output')
    parser.add_argument('--dimensions', type=str, default='2,3',
                        help='Comma-separated list of embedding dimensions (default: 2,3)')
    parser.add_argument('--taus', type=str, default='1,2,3',
                        help='Comma-separated list of time delays (default: 1,2,3)')
    return parser.parse_args()

if __name__ == "__main__":
    # Ensure the required directories exist
    for path in [
        "data/train/Bvp", "data/train/Eda", "data/train/Resp", "data/train/SpO2",
        "data/test/Bvp", "data/test/Eda", "data/test/Resp", "data/test/SpO2",
        "results", "logs"
    ]:
        os.makedirs(path, exist_ok=True)
        
    # Run the feature extraction
    args = parse_args()
    
    # Parse dimension and tau values
    dimensions = [int(d) for d in args.dimensions.split(',')]
    taus = [int(t) for t in args.taus.split(',')]
    
    run_feature_extraction(
        include_pe_verification=args.verify,
        dimensions=dimensions,
        taus=taus
    )
