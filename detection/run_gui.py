#!/usr/bin/env python3
"""
AI Video Detector GUI Launcher

This script launches the AI Video Detector GUI application.
Make sure you have installed the required dependencies:
pip install -r requirements.txt
"""

import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from index import main
    
    if __name__ == "__main__":
        print("Starting AI Video Detector GUI...")
        main()
        
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting application: {e}")
    sys.exit(1) 