"""
Brain Tumor Detection System - Streamlit Application
Deploy this app using: streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

# Check for required packages
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    st.error("❌ TensorFlow not installed. Run: pip install tensorflow==2.15.0")
    st.stop()

try:
    import cv2
except ImportError:
    st.error("❌ OpenCV not installed. Run: pip install opencv-python-headless==4.8.1.78")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .tumor-detected {
        background-color: #ffebee;
        border: 2px solid #ef5350;
    }
    .no-tumor {
        background-color: #e8f5e9;
        border: 2px solid #66bb6a;
    }
    .confidence-text {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model with detailed error handling"""
    model_path = 'brain_tumor_model.h5'
    
    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"""
        ❌ **Model file not found!**
        
        Looking for: `{model_path}`
        
        **Solutions:**
        1. Make sure you've trained the model: `python train_model.py`
        2. Check that `brain_tumor_model.h5` is in the same folder as `app.py`
        3. Current directory: `{os.getcwd()}`
        
        **Files in current directory:**
        """)
        for file in os.listdir('.'):
            st.write(f"- {file}")
        return None
    
    try:
        # Try loading the model
        with st.spinner('Loading AI model...'):
            from tensorflow import keras
            model = keras.models.load_model(model_path, compile=False)
            st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"""
        ❌ **Error loading model:**
        
        ```
        {str(e)}
        ```
        
        **Possible solutions:**
        1. Retrain the model: `python train_model.py`
        2. Check TensorFlow version: `pip install tensorflow>=2.15.0`
        3. Make sure model file isn't corrupted
        """)
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction with error handling"""
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Convert to RGB if necessary
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Resize
        img_array = cv2.resize(img_array, target_size)
        
        # Normalize
        img_array = img_array.astype('float32') # EfficientNet expects [0, 255]
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict(model, image):
    """Make prediction on image with error handling"""
    try:
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return None, None
        
        # Make prediction
        with st.spinner('Analyzing MRI scan...'):
            prediction = model.predict(processed_image, verbose=0)
        
        confidence = float(prediction[0][0])
        
        if confidence > 0.5:
            label = "Tumor Detected"
            confidence_score = confidence * 100
        else:
            label = "No Tumor"
            confidence_score = (1 - confidence) * 100
        
        return label, confidence_score
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

# Main App
def main():
    # Header
    st.markdown('<p class="main-header">🧠 Brain Tumor Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered MRI Analysis using Deep Learning</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Display system info in expander
    with st.expander("🔧 System Information"):
        st.write(f"**TensorFlow Version:** {tf.__version__}")
        st.write(f"**OpenCV Version:** {cv2.__version__}")
        st.write(f"**Python Version:** {sys.version.split()[0]}")
        st.write(f"**Current Directory:** {os.getcwd()}")
        st.write(f"**Model File Size:** {os.path.getsize('brain_tumor_model.h5') / (1024*1024):.2f} MB")
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This AI-powered system uses deep learning to detect brain tumors from MRI scans.
        
        **How it works:**
        1. Upload an MRI brain scan
        2. Click 'Detect Tumor'
        3. Get instant AI prediction
        
        **Model Details:**
        - Architecture: EfficientNetB0
        - Transfer Learning: Pre-trained on ImageNet
        - Binary Classification: Tumor / No Tumor
        """)
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This is an educational project and should NOT be used for actual medical diagnosis. 
        Always consult qualified healthcare professionals for medical advice.
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload MRI Scan")
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a brain MRI scan image in JPG or PNG format"
        )
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded MRI Scan', use_column_width=True)
                
                # Show image info
                st.info(f"📊 Image size: {image.size[0]} x {image.size[1]} pixels")
                
                # Predict button
                if st.button("🔍 Detect Tumor", type="primary", use_container_width=True):
                    # Make prediction
                    label, confidence = predict(model, image)
                    
                    if label is not None and confidence is not None:
                        # Display results in second column
                        with col2:
                            st.subheader("📊 Analysis Results")
                            
                            if label == "Tumor Detected":
                                st.markdown(f"""
                                <div class="prediction-box tumor-detected">
                                    <h2>⚠️ {label}</h2>
                                    <p class="confidence-text">Confidence: {confidence:.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                                st.error("The AI model has detected potential tumor presence in the MRI scan.")
                                st.info("**Recommendation:** Please consult a healthcare professional immediately for proper diagnosis and treatment.")
                            else:
                                st.markdown(f"""
                                <div class="prediction-box no-tumor">
                                    <h2>✅ {label}</h2>
                                    <p class="confidence-text">Confidence: {confidence:.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                                st.success("The AI model has not detected tumor presence in the MRI scan.")
                                st.info("**Note:** Regular check-ups are still recommended for maintaining good health.")
                            
                            # Additional information
                            st.markdown("---")
                            st.write("**Model Confidence Interpretation:**")
                            st.progress(confidence / 100)
                            
                            if confidence >= 90:
                                st.write("🟢 Very High Confidence")
                            elif confidence >= 75:
                                st.write("🟡 High Confidence")
                            elif confidence >= 60:
                                st.write("🟠 Moderate Confidence")
                            else:
                                st.write("🔴 Low Confidence - Consider retesting")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
        else:
            with col2:
                st.info("👈 Please upload an MRI scan image to begin analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Developed for Educational Purposes | Deep Learning & AI Project</p>
        <p>Model: EfficientNetB0 with Transfer Learning | Framework: TensorFlow & Keras</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()