# mnist_cnn.py (Optimized PyTorch Version)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load dataset
print("Loading MNIST dataset...")
train_data = datasets.MNIST(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

# CNN Model definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64*7*7)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model, loss, and optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with progress feedback
def train_model(epochs=10):
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        # Use tqdm for progress bar
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            epoch_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}", 
                             acc=f"{100 * correct/total:.2f}%")
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, Time: {epoch_time:.1f}s")
        start_time = time.time()
    
    print("Training complete!")

# Evaluation function
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy >= 95  # Return True if >95% accuracy

# Train and evaluate
print("Starting training...")
train_model(epochs=10)  # Train for 10 epochs

print("\nEvaluating model...")
accuracy_achieved = evaluate_model()

# Visualization of predictions
def visualize_predictions(num_samples=5):
    model.eval()
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    with torch.no_grad():
        # Get random samples
        indices = np.random.choice(len(test_data), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            image, label = test_data[idx]
            
            # Predict
            input_tensor = image.unsqueeze(0).to(device)
            output = model(input_tensor)
            _, prediction = torch.max(output, 1)
            
            # Plot
            axes[i].imshow(image.squeeze(), cmap='gray')
            axes[i].set_title(f'Pred: {prediction.item()}, True: {label}')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_predictions.png')
    print("Sample predictions saved as 'mnist_predictions.png'")

# Visualize predictions
print("\nGenerating sample predictions...")
visualize_predictions()

# Save model if accuracy target achieved
if accuracy_achieved:
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("Model saved successfully as 'mnist_cnn.pth'")
else:
    print("Accuracy target not achieved. Consider training for more epochs or tuning hyperparameters")

print("\nAll tasks completed!")