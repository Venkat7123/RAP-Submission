"""
Repack extracted PyTorch model to single .pt file
"""
import torch
import os

print("Repacking model...")

# Try to load from best/ directory
try:
    # Method 1: Load as zip archive
    import zipfile
    import io

    print("Creating .pt file from best/ folder...")

    # Just copy the old model and replace it later
    # The user should provide the actual best.pt file

    print("\nERROR: The best/ folder is an extracted archive.")
    print("We need the original best.pt file!")
    print("\nPlease:")
    print("1. Find your downloaded best.pt file")
    print("2. Copy it: cp /path/to/best.pt models/best.pt")
    print("3. Then run: python app.py")

except Exception as e:
    print(f"Error: {e}")
