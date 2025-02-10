# Image Classification API using FastAPI and PyTorch

This project implements a REST API for image classification using FastAPI and PyTorch. The model used is a pre-trained ResNet-18 model, fine-tuned for CIFAR-10 image classification. The API accepts images via HTTP requests and returns the predicted class along with the confidence score.

## Table of Contents

- [Overview](#overview)
- [Project Setup](#project-setup)
- [Model Details](#model-details)
- [API Endpoints](#api-endpoints)
  - [Root Endpoint](#root-endpoint)
  - [Prediction Endpoint](#prediction-endpoint)
- [Testing with Postman](#testing-with-postman)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This project provides an API service for classifying images using a pre-trained Convolutional Neural Network (CNN) model, specifically the ResNet-18 model. The FastAPI framework is used for the API, and PyTorch is used to load and run the model. 

The API accepts an image file via a `POST` request to the `/predict` endpoint, processes it using the model, and returns the predicted class label along with a confidence score.

### Technologies Used

- **FastAPI**: A fast and modern web framework for building APIs with Python 3.7+.
- **PyTorch**: A machine learning framework for developing and running neural networks.
- **ResNet-18**: A pre-trained CNN model used for image classification.
- **ngrok**: To expose the local server to the internet (for testing purposes).
- **Postman**: For testing the API.

## Project Setup

### Prerequisites

Ensure that you have the following installed:

- Python 3.7+
- Pip
- PyTorch (including torchvision)
- FastAPI
- Uvicorn
- Ngrok (for public URL access)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone (https://github.com/kavyaLokuge/cifar_10_image_classification.git)
   cd cifar_10_image_classification

## Project Setup

### Create a virtual environment (Optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the FastAPI app:

```bash
uvicorn app:app --reload
```

This will start the FastAPI server at `http://localhost:8000`.

### Set up ngrok to expose the local server to the internet (optional):

1. Download and install ngrok: [https://ngrok.com/download](https://ngrok.com/download)
2. Run the following command to expose port `8000`:

```bash
ngrok http 8000
```

You will receive a public URL to test your API externally.

## Model Details

The project uses ResNet-18, a deep residual network pre-trained on the CIFAR-10 dataset. ResNet-18 is known for its ability to perform well on image classification tasks.

### Model Summary:
- **Architecture**: ResNet-18
- **Dataset**: CIFAR-10
- **Classes**:
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

### Pre-trained Weights:
The model uses pre-trained weights from the CIFAR-10 dataset, which helps to classify the images efficiently.

## API Endpoints

### Root Endpoint

- **URL**: `/`
- **Method**: GET
- **Description**: Returns a welcome message to confirm the API is running.

**Response Example**:
```json
{
  "message": "Welcome to the CIFAR-10 Image Classification API. Use /predict to classify images."
}
```

### Prediction Endpoint

- **URL**: `/predict`
- **Method**: POST
- **Description**: Accepts an image file and returns the predicted class and confidence score.

#### Request Body:
Use Postman or another API client to send an image as `multipart/form-data` under the key `file`.

**Example Request (Postman)**:
- Select `POST` method.
- URL: `http://localhost:8000/predict` (or the ngrok public URL).
- Under `Body`, choose `form-data`.
- Add a key `file` and select an image file to upload.

**Response Example**:
```json
{
  "class": "automobile",
  "confidence": 0.2576049566268921
}
```

## Testing with Postman

To test the API using Postman:

1. Open Postman and create a new POST request.
2. Enter the URL for your local or ngrok endpoint (e.g., `http://localhost:8000/predict` or the ngrok URL).
3. In the Body section, choose `form-data`.
4. Add a new key named `file`, and select an image to upload.
5. Send the request and inspect the response.

### Postman Response Screenshot
Here is an example of the response you should get when testing the API:

![Postman Response]([path/to/your/screenshot.png](https://drive.google.com/file/d/15UIyFRNjtapcSg2hkmiAcXXwtfU68hn_/view?usp=sharing))

This image shows a sample response where the predicted class is "automobile" with a confidence score of approximately `0.2576`.

## Deployment

The application can be deployed on any platform supporting FastAPI, such as:

- **Docker**: Package your application using Docker and deploy it to cloud services like AWS, Azure, or Google Cloud.
- **Heroku**: Host the app with free or paid dynos on Heroku.
- **AWS Lambda**: For serverless deployment, using AWS Lambda functions in combination with API Gateway.

### Docker Setup

To containerize the application using Docker:

1. **Build the Docker image**:

```bash
docker build -t image-classification-api .
```

2. **Run the Docker container**:

```bash
docker run -d -p 8000:8000 image-classification-api
```

Access the API at `http://localhost:8000`.

## Troubleshooting

### Error: Connection refused
- Make sure the FastAPI server is running on `localhost:8000`.
- If using Docker or ngrok, ensure that ports are properly exposed.

### Error: Missing file field
- Ensure that you're sending the image correctly using `multipart/form-data` in the `file` field.

For more information on troubleshooting, refer to the FastAPI documentation.

