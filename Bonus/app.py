# Bonus/app.py
import os
import urllib.request
import numpy as np
import streamlit as st
from PIL import Image
import cv2

# Set page config
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🔢", layout="wide")

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
        # Load image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        # Resize and normalize
        img = cv2.resize(img, (28, 28))
        img = img.astype(np.float32) / 255.0
        
        # Display the processed image
        st.image(img, caption="Uploaded Image", width=200)
        
        # Check if model exists
        MODEL_PATH = "mnist_cnn.pth"
        if not os.path.exists(MODEL_PATH):
            with st.spinner('Downloading pre-trained model...'):
                try:
                    urllib.request.urlretrieve(
                        "https://github.com/rasbt/mnist-cnn-pytorch/raw/main/mnist_cnn.pth", 
                        MODEL_PATH
                    )
                except Exception as e:
                    st.error(f"Model download failed: {str(e)}")
                    st.stop()
        
        # Import torch only after model download
        import torch
        from torchvision import transforms
        
        # Model definition
        class SimpleCNN(torch.nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
                self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
                self.fc1 = torch.nn.Linear(320, 50)
                self.fc2 = torch.nn.Linear(50, 10)
                
            def forward(self, x):
                x = torch.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
                x = torch.relu(torch.nn.functional.max_pool2d(self.conv2(x), 2))
                x = x.view(-1, 320)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return torch.nn.functional.log_softmax(x, dim=1)
        
        # Load model
        model = SimpleCNN()
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        
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
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png", 
             caption="MNIST Examples", width=300)
    
    # Add download link
    st.markdown("### Download Sample Images")
    st.markdown("[MNIST Test Images](https://github.com/pytorch/hub/raw/master/images/mnist.png)")

# Add footer
st.markdown("---")
st.markdown("### ℹ️ Instructions:")
st.markdown("1. Upload an image of a handwritten digit (0-9)")
st.markdown("2. The image will be resized to 28x28 pixels and converted to grayscale")
st.markdown("3. The model will predict the digit with confidence scores")
st.markdown("4. Try the sample images on the right for testing")

# Add GitHub link
st.markdown("[View source code on GitHub](https://github.com/yourusername/ml-suite)")

# Add note about model
st.info("**Note:** This app uses a pre-trained CNN model trained on the MNIST dataset")