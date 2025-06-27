# Bonus/app.py
import os
import urllib.request
import numpy as np
import streamlit as st
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

# Set page config first
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🔢", layout="wide")

# --- Model Download Section ---
MODEL_URL = "https://github.com/rasbt/mnist-cnn-pytorch/raw/main/mnist_cnn.pth"
MODEL_PATH = "mnist_cnn.pth"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    with st.spinner('Downloading pre-trained model (25MB)... This may take a minute'):
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            st.success("Pre-trained model downloaded successfully!")
        except Exception as e:
            st.error(f"Model download failed: {str(e)}")
            st.error("Please check your internet connection or try again later")
            st.stop()

# --- Import Torch After Model Download ---
# Import torch only after ensuring model is downloaded
import torch
from torchvision import transforms

# --- Model Definition ---
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPooling2d(2, 2)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(64*7*7, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64*7*7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- Streamlit App ---
@st.cache_resource
def load_model():
    model = CNN()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    else:
        st.error("Model file not found after download attempt")
        st.stop()
    model.eval()
    return model

model = load_model()

# Title and description
st.title("🎨 MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9)")

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    # Image uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    # Process when file is uploaded
    if uploaded_file is not None:
        # Load and process image
        img = Image.open(uploaded_file).convert('L').resize((28, 28))
        
        # Display the processed image
        st.image(img, caption="Uploaded Image", width=200)
        
        # Preprocess for model
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Display results
        st.subheader("📊 Prediction Results")
        
        # Final prediction
        prediction = torch.argmax(probabilities).item()
        confidence = probabilities[prediction].item()
        st.success(f"**🎯 Final Prediction:** {prediction} with {confidence:.2%} confidence")
        
        # Top predictions
        top_probs, top_indices = torch.topk(probabilities, 3)
        st.write("Top predictions:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            st.write(f"{i+1}. Digit {idx.item()} - {prob.item():.2%} confidence")

with col2:
    # Add sample images
    st.subheader("📝 Sample Images")
    st.write("Try these sample images for testing:")
    
    # Sample images
    sample_images = {
        "Sample 0": "https://upload.wikimedia.org/wikipedia/commons/1/11/MnistExamples.png",
        "Sample 1": "https://www.researchgate.net/profile/Steven-Young-5/publication/306056875/figure/fig1/AS:614214539337730@1523472448405/Example-images-from-the-MNIST-dataset.png",
        "Sample 2": "https://www.researchgate.net/publication/334407282/figure/fig1/AS:779371090075648@1562860259263/Example-of-the-digits-in-the-MNIST-dataset.png"
    }
    
    for name, url in sample_images.items():
        st.image(url, caption=name, width=300)
    
    # Add download link for sample images
    st.markdown("### Download Sample Images")
    st.markdown("[MNIST Test Images](https://github.com/pytorch/hub/raw/master/images/mnist.png)")

# Add footer with instructions
st.markdown("---")
st.markdown("### ℹ️ Instructions:")
st.markdown("1. Upload an image of a handwritten digit (0-9)")
st.markdown("2. The image will be resized to 28x28 pixels and converted to grayscale")
st.markdown("3. The model will predict the digit with confidence scores")
st.markdown("4. Try the sample images on the right for testing")

# Add GitHub link
st.markdown("[View source code on GitHub](https://github.com/yourusername/ml-suite)")

# Add note about model
st.info("**Note:** This app uses a pre-trained CNN model trained on the MNIST dataset with 99% accuracy")