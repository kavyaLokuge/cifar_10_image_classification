# Import necessary modules
from fastapi import FastAPI, File, UploadFile
import uvicorn
import torch
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import io
import json
from pyngrok import ngrok, conf
import nest_asyncio
import os

# Apply asyncio to run the FastAPI app in Colab
nest_asyncio.apply()

# Define class labels for CIFAR-10 dataset
class_labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Define transformations for image preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit ResNet input
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Load and preprocess the CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms)

# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load the pre-trained ResNet model and modify the final layer for CIFAR-10 classification
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)  # Adjust for CIFAR-10 classification

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fine-tune the model (optional code for training)
def fine_tune_model():
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    print("Model fine-tuned")

# Uncomment to fine-tune the model
#fine_tune_model()

# Function to make predictions
def predict(image_bytes):
    model.eval()
    image = Image.open(io.BytesIO(image_bytes))
    image = data_transforms(image).unsqueeze(0).to(device)  # Apply transformations
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs[0], dim=0)  # Compute softmax probabilities
        predicted_class = torch.argmax(probabilities).item()  # Get the class index
    return {"class": class_labels[predicted_class], "confidence": probabilities[predicted_class].item()}  # Return human-readable label

# Initialize FastAPI app
app = FastAPI()

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the CIFAR-10 Image Classification API. Use /predict to classify images."}

# Prediction endpoint
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict(image_bytes)
    return json.dumps(result)

# Docker and cloud configurations
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "YOUR AUTH TOKEN") #Replace this with your ngrok auth token
conf.get_default().auth_token = NGROK_AUTH_TOKEN

# Set up ngrok to expose FastAPI app on an external URL
public_url = ngrok.connect(8000)
print(f"FastAPI app is publicly accessible at: {public_url}")

# Run FastAPI application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)