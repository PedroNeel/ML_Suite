# app.py (Fixed Import Structure)
import os
import sys
import subprocess
import urllib.request

# --- Package Installation Section (No Streamlit usage here) ---
try:
    # Try importing required packages
    import numpy as np
    import torch
    from torchvision import transforms
    from PIL import Image
except ImportError:
    # Install missing packages without using Streamlit
    print("Installing required packages...")
    requirements = [
        "numpy", 
        "torch", 
        "torchvision", 
        "Pillow"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + requirements)
    
    # Re-import after installation
    import numpy as np
    import torch
    from torchvision import transforms
    from PIL import Image

# Now safely import Streamlit and canvas
try:
    import streamlit as st
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    print("Installing Streamlit and canvas...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "streamlit", "streamlit-drawable-canvas"])
    import streamlit as st
    from streamlit_drawable_canvas import st_canvas

# --- Model Download Section ---
MODEL_URL = "https://github.com/rasbt/mnist-cnn-pytorch/raw/main/mnist_cnn.pth"
MODEL_PATH = "mnist_cnn.pth"

# Download model if not exists (using Streamlit now)
if not os.path.exists(MODEL_PATH):
    st.info("Downloading pre-trained model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Pre-trained model downloaded successfully!")
    except Exception as e:
        st.error(f"Model download failed: {str(e)}")
        st.stop()

# --- Model Definition ---
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
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

st.title("🎨 MNIST Digit Classifier")
st.write("Draw a digit (0-9) in the canvas below")

# Create a drawing canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=15,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Process when something is drawn
if canvas_result.image_data is not None:
    # Convert canvas data to image
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
    
    # Convert to grayscale and resize
    img = img.convert('L').resize((28, 28))
    
    # Display the processed image
    st.image(img, caption="Processed Input", width=100)

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
    
    # Create two columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence bar chart
        st.bar_chart(probabilities.numpy())
    
    with col2:
        # Top predictions table
        top_probs, top_indices = torch.topk(probabilities, 3)
        predictions = [
            {"Digit": str(i.item()), "Confidence": f"{p.item():.2%}"} 
            for p, i in zip(top_probs, top_indices)
        ]
        st.table(predictions)
    
    # Final prediction
    prediction = torch.argmax(probabilities).item()
    confidence = probabilities[prediction].item()
    st.success(f"**🎯 Final Prediction:** {prediction} with {confidence:.2%} confidence")
    
    # Add clear button
    if st.button("Clear Canvas"):
        st.experimental_rerun()
else:
    st.info("Draw a digit in the canvas to see predictions")

# Add footer with instructions
st.markdown("---")
st.markdown("### 📝 Tips for Best Results:")
st.markdown("- Draw a single digit in the center of the canvas")
st.markdown("- Make strokes thick and clear")
st.markdown("- Avoid touching the edges of the canvas")
st.markdown("- Click 'Clear Canvas' to start over")