@echo off
echo ================================================
echo Installing Brain Tumor Detection Dependencies
echo ================================================
echo.

echo Step 1: Upgrading pip...
python -m pip install --upgrade pip
echo.

echo Step 2: Uninstalling old OpenCV versions...
python -m pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y
echo.

echo Step 3: Installing TensorFlow...
python -m pip install tensorflow==2.15.0
echo.

echo Step 4: Installing Streamlit...
python -m pip install streamlit==1.29.0
echo.

echo Step 5: Installing OpenCV (headless)...
python -m pip install opencv-python-headless==4.8.1.78
echo.

echo Step 6: Installing other dependencies...
python -m pip install numpy==1.24.3
python -m pip install pillow==10.1.0
python -m pip install matplotlib==3.8.2
python -m pip install scikit-learn==1.3.2
echo.

echo ================================================
echo Verifying installations...
echo ================================================
echo.

echo Checking TensorFlow...
python -c "import tensorflow as tf; print('✓ TensorFlow:', tf.__version__)"
echo.

echo Checking Streamlit...
python -c "import streamlit as st; print('✓ Streamlit:', st.__version__)"
echo.

echo Checking OpenCV...
python -c "import cv2; print('✓ OpenCV:', cv2.__version__)"
echo.

echo Checking NumPy...
python -c "import numpy as np; print('✓ NumPy:', np.__version__)"
echo.

echo Checking Pillow...
python -c "from PIL import Image; print('✓ Pillow installed')"
echo.

echo ================================================
echo Installation complete!
echo You can now run: streamlit run app.py
echo ================================================
pause