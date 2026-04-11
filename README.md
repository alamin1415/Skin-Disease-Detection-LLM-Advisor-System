# Skin Disease Classification using Deep Learning (VGG16)

## Project Overview

This project builds a deep learning model to classify different types of skin
diseases using image data. The model is based on **transfer learning with
VGG16**, trained on a labeled skin disease dataset.

---

## Dataset

- Source: Kaggle skin disease image dataset
- Classes: Multiple skin disease categories (e.g., Atopic Dermatitis, Melanoma,
  Psoriasis, etc.)
- Dataset Structure: Images organized into class-based folders

---

## Data Preprocessing

### 1. Dataset Download

- Downloaded dataset using Kaggle API
- Extracted and loaded images into Colab environment

### 2. Train/Validation/Test Split

- Train: 70%
- Validation: 15%
- Test: 15%

Each class folder was split manually using Python scripts.

### 3. Image Processing

- Image size resized to: 224 × 224
- Normalization applied (rescale 1/255)
- Data augmentation applied:
  - Rotation
  - Zoom
  - Horizontal flip

---

## Model Architecture

### Transfer Learning (VGG16)

- Base model: VGG16 (pretrained on ImageNet)
- Top layers removed (`include_top=False`)
- All base layers frozen (non-trainable)

### Custom Classifier Head

- Flatten layer
- Dense layer (256 neurons, ReLU activation)
- Output layer (Softmax activation for multi-class classification)

---

## Model Compilation

- Loss function: Categorical Crossentropy
- Optimizer: Adam
- Metrics: Accuracy

---

## Training Process

- Batch size: 32
- Epochs: 10
- Steps per epoch: Based on dataset size
- Validation used during training
- Class imbalance handled using class weights

---

## Evaluation

### Model Performance on Test Data

The trained model was evaluated on the test dataset and achieved the following
results:

Test Loss: 0.9472 Test Accuracy: 0.6621 (66.21%)

## AI Recommendation API (FastAPI + Ollama LLM)

This project includes a **FastAPI-based backend API** that combines:

- A trained deep learning model (VGG16) for skin disease prediction
- A local LLM (Ollama - Qwen2.5) for generating medical-style explanations

---

## API Workflow

### 1. Image Upload

- User sends an image to the API endpoint:

POST /analyze_skin

---

### 2. Skin Disease Prediction (ML Model)

- The uploaded image is processed using a trained CNN model
- The model returns:
  - Predicted disease class
  - Confidence score

---

### 3. LLM-Based Recommendation (Ollama)

- The predicted result is passed to a local LLM (`qwen2.5:3b`)
- The LLM generates a patient-friendly medical explanation including:
  1. Simple explanation of the disease
  2. Possible causes
  3. Home care instructions
  4. When to consult a doctor

---

## Response Format

The API returns a combined JSON response:

````json
{
  "disease": "eczema",
  "confidence": 0.93,
  "recommendations": "Mild inflammatory skin condition caused by irritation or allergic reaction.",
  "next_steps": "Consult a dermatologist if symptoms persist or worsen.",
  "tips": "Keep skin moisturized, avoid harsh chemicals, and maintain hygiene."
}
``` id="r2m9xq"

---

## Tech Stack
- FastAPI (Backend API)
- TensorFlow / Keras (ML Model)
- VGG16 (Transfer Learning)
- Ollama (Local LLM Server)
- Qwen2.5:3b Model
- Pillow (Image Processing)
- Uvicorn (Server)

---

## Key Features
- Real-time image-based disease prediction
- AI-generated medical explanations
- Local LLM (no external API required)
- Fast and lightweight REST API
- Combines Computer Vision + NLP

---

## Endpoint Details

### `POST /analyze_skin`
Uploads an image and returns:
- Predicted disease
- Confidence score
- AI-generated recommendation

---

## System Architecture


User Image
↓
FastAPI Server
↓
VGG16 Model (Disease Prediction)
↓
Ollama LLM (Qwen2.5)
↓
Final JSON Response




````
