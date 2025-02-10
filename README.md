# AI-powered application for Image Classification using PyTorch and the ResNet model with CIFAR-10 dataset. (Interview Assignment)

This project is an AI-powered application that uses a pre-trained ResNet-18 model fine-tuned on the CIFAR-10 dataset to classify images into one of 10 categories. The application is built using PyTorch for the model and FastAPI for the API. It is designed to be deployed locally or on Google Colab using ngrok.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Fine-Tuning the Model](#fine-tuning-the-model)
- [Deployment](#deployment)
- [Future Improvements](#future-improvements)

## Project Overview

The goal of this project is to build an image classification system using PyTorch and FastAPI. The system uses a pre-trained ResNet-18 model, fine-tuned on the CIFAR-10 dataset, to classify images into one of the following 10 categories:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The application exposes an API endpoint that accepts an image file and returns the predicted class along with the confidence score.

## Requirements

To run this project, you need the following dependencies:

- Python 3.8 or higher
- PyTorch
- FastAPI
- Uvicorn
- Pillow
- Pyngrok
- Nest-asyncio

You can install all the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Setup Instructions

1. Clone this repository:

```bash
git clone https://github.com/kavyaLokuge/cifar_10_image_classification.git
cd cifar10-image-classification
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the CIFAR-10 dataset (this will be done automatically when you run the script for the first time).

## Running the Application

To run the application locally, follow these steps:

1. Start the FastAPI server:

```bash
python image_classification.py
```

The application will start on `http://0.0.0.0:8000`. You can access the API documentation at `http://0.0.0.0:8000/docs`.

Important: If you want to expose the API publicly (e.g., for testing on Google Colab), you need to update your ngrok authentication. To do this, run the following command to authenticate ngrok:
```bash
ngrok authtoken <your-ngrok-auth-token>
```
This will link ngrok to your account and allow the script to generate a public URL for your API. The script will then automatically set up ngrok to provide a public URL.

## API Endpoints

### 1. Root Endpoint

- **URL**: `/`
- **Method**: `GET`
- **Description**: Returns a welcome message.
  
**Response**:

```json
{
  "message": "Welcome to the CIFAR-10 Image Classification API. Use /predict to classify images."
}
```

### 2. Prediction Endpoint

- **URL**: `/predict`
- **Method**: `POST`
- **Description**: Accepts an image file and returns the predicted class and confidence score.
  
**Request Body**:
- `file`: Image file (JPEG, PNG, etc.)

**Response**:

```json
{
  "class": "cat",
  "confidence": 0.95
}
```

## Fine-Tuning the Model

The ResNet-18 model is fine-tuned on the CIFAR-10 dataset using the `fine_tune_model` function. By default, this function is commented out in the script. To fine-tune the model, follow these steps:

1. Uncomment the following line in `image_classification.py`:

```python
#fine_tune_model()
```

2. Run the script again. The model will be fine-tuned for 5 epochs, and the fine-tuned weights will be used for predictions.

## Deployment

### Local Deployment

To deploy the application locally, ensure you have Docker installed and follow these steps:

1. Build the Docker image:

```bash
docker build -t cifar10-image-classification .
```

2. Run the Docker container:

```bash
docker run -p 8000:8000 cifar10-image-classification
```

3. Access the API at `http://localhost:8000`.

### Cloud Deployment

To deploy the application to a cloud service (e.g., AWS Lambda, Hugging Face Spaces, or Google Cloud), you can use the provided Dockerfile and follow the deployment instructions for your chosen platform.
```
