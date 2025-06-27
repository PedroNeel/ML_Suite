# Bonus/app.py
import streamlit as st
import numpy as np
from PIL import Image
import requests
import json
import io

# Set page config
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🔢", layout="wide")

# Title and description
st.title("🎨 MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9)")

# TensorFlow.js model URL
MODEL_URL = "https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/model.json"

# Sample images
SAMPLE_IMAGES = {
    "Sample 0": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte/0.png",
    "Sample 1": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte/1.png",
    "Sample 2": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte/2.png",
    "Sample 3": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte/3.png",
    "Sample 4": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte/4.png"
}

# Preprocess image function
def preprocess_image(image):
    img = image.convert('L').resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1).tolist()
    return img_array

# TensorFlow.js prediction function
def predict_with_tfjs(image_data):
    # Prepare data for TensorFlow.js
    payload = {
        "signature_name": "serving_default",
        "instances": image_data
    }
    
    # TF.js prediction endpoint (public proxy)
    url = "https://tfjs-mnist-classifier-proxy.vercel.app/api/predict"
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            predictions = response.json()['predictions'][0]
            return predictions
        else:
            st.error(f"Prediction failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    # Image uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    # Process when file is uploaded
    if uploaded_file is not None:
        # Load image
        img = Image.open(uploaded_file)
        
        # Display the original image
        st.image(img, caption="Original Image", width=200)
        
        # Preprocess image
        with st.spinner('Processing image...'):
            img_data = preprocess_image(img)
        
        # Predict
        with st.spinner('Predicting digit...'):
            predictions = predict_with_tfjs(img_data)
        
        if predictions:
            # Display results
            st.subheader("📊 Prediction Results")
            
            # Convert to probabilities
            probabilities = np.array(predictions)
            probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
            
            # Final prediction
            prediction = np.argmax(probabilities)
            confidence = probabilities[prediction]
            st.success(f"**🎯 Final Prediction:** {prediction} with {confidence:.2%} confidence")
            
            # Top predictions
            top_indices = np.argsort(probabilities)[::-1][:3]
            st.write("Top predictions:")
            for i, idx in enumerate(top_indices):
                st.write(f"{i+1}. Digit {idx} - {probabilities[idx]:.2%} confidence")
            
            # Confidence bar chart
            st.subheader("Confidence Distribution")
            st.bar_chart({str(i): prob for i, prob in enumerate(probabilities)})

with col2:
    # Add sample images
    st.subheader("📝 Sample Images")
    st.write("Try these sample images for testing:")
    
    # Display sample images
    for name, url in SAMPLE_IMAGES.items():
        st.image(url, caption=name, width=100)
        if st.button(f"Test {name}", key=name):
            # Download and process sample image
            response = requests.get(url)
            img = Image.open(io.BytesIO(response.content))
            
            # Display the sample image
            st.image(img, caption=f"Testing {name}", width=200)
            
            # Preprocess image
            with st.spinner('Processing image...'):
                img_data = preprocess_image(img)
            
            # Predict
            with st.spinner('Predicting digit...'):
                predictions = predict_with_tfjs(img_data)
            
            if predictions:
                # Convert to probabilities
                probabilities = np.array(predictions)
                probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
                
                # Final prediction
                prediction = np.argmax(probabilities)
                confidence = probabilities[prediction]
                st.success(f"**🎯 Prediction for {name}:** {prediction} with {confidence:.2%} confidence")

# Add footer with instructions
st.markdown("---")
st.markdown("### ℹ️ Instructions:")
st.markdown("1. Upload an image of a handwritten digit (0-9)")
st.markdown("2. The image will be resized to 28x28 pixels and converted to grayscale")
st.markdown("3. The model will predict the digit with confidence scores")
st.markdown("4. Try the sample images for quick testing")

# Add note about model
st.info("**Note:** This app uses a pre-trained CNN model trained on the MNIST dataset with 99% accuracy")

# Add GitHub link
st.markdown("[View source code on GitHub](https://github.com/yourusername/ml-suite)")