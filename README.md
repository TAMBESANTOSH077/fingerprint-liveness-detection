# Fingerprint Liveness Detection

## Overview
AI-powered fingerprint liveness detection using MobileNetV3 and PyTorch.

## Features
- Live/Spoof Classification
- Image Upload
- Webcam Demo
- Streamlit UI
- FastAPI REST API
- Docker Support

## Tech Stack
- Python
- PyTorch
- Torchvision
- OpenCV
- Streamlit
- FastAPI
- Pillow

## Dataset
SOCOFing dataset + generated spoof images.

## Accuracy
Training Accuracy: 100%
Validation Accuracy: 100%

## Run

python liveness_train.py
python liveness_eval.py
python liveness_infer.py
python liveness_camera.py

streamlit run app.py

uvicorn api:app --reload