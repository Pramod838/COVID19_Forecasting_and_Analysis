#!/usr/bin/env python3
"""
Script to download all required datasets.
Run this first before analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.downloaders import main

if __name__ == "__main__":
    print("="*60)
    print("COVID-19 Disease Dynamics Analysis - Data Download")
    print("="*60)
    
    try:
        main()
        print("\n✓ All datasets downloaded successfully!")
        print("\nNext steps:")
        print("  1. Run: jupyter notebook")
        print("  2. Open notebooks/01_eda_covid_data.ipynb")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
