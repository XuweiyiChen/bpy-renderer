#!/usr/bin/env python3
"""
Simple launcher for batch rendering Objaverse GLB files.
"""

import os
import sys
import subprocess

def main():
    # Check if we're in the right directory
    if not os.path.exists("examples/object/render_objaverse_batch.py"):
        print("Error: Please run this script from the bpy-renderer root directory")
        sys.exit(1)
    
    # Check if conda environment is activated
    if "CONDA_DEFAULT_ENV" not in os.environ:
        print("Warning: No conda environment detected")
        print("Please activate your conda environment first:")
        print("conda activate /home/ubuntu/xuweiyi/bpy-renderer/env")
        print()
    
    # Install requirements if needed
    print("Installing additional requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "batch_render_requirements.txt"])
        print("Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to install requirements: {e}")
        print("Please install manually: pip install opencv-python joblib numpy")
    
    # Run the batch rendering script
    print("\nStarting batch rendering...")
    print("="*60)
    
    try:
        # Change to the examples/object directory
        os.chdir("examples/object")
        
        # Run the batch rendering script
        subprocess.check_call([sys.executable, "render_objaverse_batch.py"])
        
        print("="*60)
        print("Batch rendering completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Batch rendering failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nBatch rendering interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main() 