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
   git clone https://github.com/your-repository-url.git
   cd your-project-folder
