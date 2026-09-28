#!/bin/bash
echo "Installing required dependencies..."
python3 -m pip install numpy librosa scipy
echo "Dependencies installed successfully!"
echo "To run the audio identification system, use: python3 run_demo.py"
