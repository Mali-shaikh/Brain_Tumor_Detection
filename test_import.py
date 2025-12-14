"""
Debug script to test imports
"""
import sys

print("=" * 60)
print("Python Environment Check")
print("=" * 60)
print(f"\nPython Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"\nPython Path:")
for path in sys.path:
    print(f"  - {path}")

print("\n" + "=" * 60)
print("Testing Package Imports")
print("=" * 60)

# Test TensorFlow
try:
    import tensorflow as tf
    print(f"\n✓ TensorFlow: {tf.__version__}")
    print(f"  Location: {tf.__file__}")
except Exception as e:
    print(f"\n✗ TensorFlow Error: {e}")

# Test Keras
try:
    from tensorflow import keras
    print(f"\n✓ Keras: {keras.__version__}")
    print(f"  Location: {keras.__file__}")
except Exception as e:
    print(f"\n✗ Keras Error: {e}")

# Test Streamlit
try:
    import streamlit as st
    print(f"\n✓ Streamlit: {st.__version__}")
    print(f"  Location: {st.__file__}")
except Exception as e:
    print(f"\n✗ Streamlit Error: {e}")

# Test OpenCV
try:
    import cv2
    print(f"\n✓ OpenCV: {cv2.__version__}")
    print(f"  Location: {cv2.__file__}")
except Exception as e:
    print(f"\n✗ OpenCV Error: {e}")

# Test NumPy
try:
    import numpy as np
    print(f"\n✓ NumPy: {np.__version__}")
    print(f"  Location: {np.__file__}")
except Exception as e:
    print(f"\n✗ NumPy Error: {e}")

# Test PIL
try:
    from PIL import Image
    import PIL
    print(f"\n✓ Pillow: {PIL.__version__}")
    print(f"  Location: {PIL.__file__}")
except Exception as e:
    print(f"\n✗ Pillow Error: {e}")

print("\n" + "=" * 60)
print("Testing Model Loading")
print("=" * 60)

try:
    import os
    model_path = 'brain_tumor_model.keras'
    if os.path.exists(model_path):
        print(f"\n✓ Model file exists: {model_path}")
        print(f"  Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        from tensorflow import keras
        print("\n  Attempting to load model...")
        model = keras.models.load_model(model_path, compile=False)
        print("  ✓ Model loaded successfully!")
    else:
        print(f"\n✗ Model file not found: {model_path}")
except Exception as e:
    print(f"\n✗ Model Loading Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All checks complete!")
print("=" * 60)