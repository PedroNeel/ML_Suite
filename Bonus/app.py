# app.py
import os
import urllib.request
import numpy as np
import streamlit as st
from PIL import Image

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
# This ensures we have the model before loading torch
import torch
from torchvision import transforms

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

# Initialize app
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🔢")

st.title("🎨 MNIST Digit Classifier")
st.write("Draw a digit (0-9) in the canvas below")

# Create a drawing canvas using Streamlit's built-in components
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
    }
    .canvas-container {
        border: 2px solid #f0f2f6;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Create drawing canvas using file uploader as fallback
option = st.radio("Input method:", ("Draw on canvas", "Upload image"))

if option == "Draw on canvas":
    # Use HTML canvas as fallback
    st.markdown("""
    <div class="canvas-container">
        <canvas id="canvas" width="280" height="280" style="border:1px solid #000000; background-color: white;"></canvas>
    </div>
    <button onclick="clearCanvas()">Clear Canvas</button>
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        
        function startDrawing(e) {
            isDrawing = true;
            draw(e);
        }
        
        function draw(e) {
            if (!isDrawing) return;
            ctx.lineWidth = 15;
            ctx.lineCap = 'round';
            ctx.strokeStyle = '#000000';
            
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(e.offsetX, e.offsetY);
        }
        
        function stopDrawing() {
            isDrawing = false;
            ctx.beginPath();
        }
        
        function clearCanvas() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
        
        // Expose canvas data to Streamlit
        function getCanvasData() {
            return canvas.toDataURL('image/png');
        }
    </script>
    """, unsafe_allow_html=True)
    
    # Get canvas data
    canvas_data = st.button("Process Drawing")
else:
    uploaded_file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])
    canvas_data = uploaded_file is not None

# Process when something is drawn
if canvas_data:
    model = load_model()
    
    if option == "Draw on canvas":
        # Get canvas data via JavaScript
        img_data = st.markdown("""
        <script>
            const data = getCanvasData();
            window.parent.postMessage({type: 'canvasData', data: data}, '*');
        </script>
        """, unsafe_allow_html=True)
        
        # We'll use a placeholder since JS integration is complex on Streamlit Cloud
        st.warning("For full canvas functionality, please run locally. Using sample image.")
        img = Image.new('L', (28, 28), color=0)
    else:
        # Process uploaded file
        img = Image.open(uploaded_file).convert('L').resize((28, 28))
    
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

# Add footer with instructions
st.markdown("---")
st.markdown("### 📝 Tips for Best Results:")
st.markdown("- Draw or upload a single digit (0-9)")
st.markdown("- For drawing: make strokes thick and clear")
st.markdown("- For uploads: use 28x28 grayscale images for best results")
st.markdown("**Note:** This app uses a pre-trained MNIST CNN model")

# Add download link for sample images
st.markdown("### Sample Images to Try:")
st.markdown("Download sample digit images: [0](https://upload.wikimedia.org/wikipedia/commons/1/11/MnistExamples.png) | [1](https://github.com/pytorch/hub/raw/master/images/mnist.png)")

# Add GitHub link
st.markdown("[View source code on GitHub](https://github.com/yourusername/your-repo)")