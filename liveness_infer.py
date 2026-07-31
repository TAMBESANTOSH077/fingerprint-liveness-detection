import os
import sys
import glob
import random

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==========================================
# Device
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

# ==========================================
# Model
# ==========================================

model = models.mobilenet_v3_small(weights=None)

in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, 2)

model.load_state_dict(
    torch.load(
        "models/liveness_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("Model Loaded Successfully")

# ==========================================
# Image Selection
# ==========================================

if len(sys.argv) > 1:
    image_path = os.path.normpath(sys.argv[1])
else:
    IMAGE_FOLDER = "dataset_split/test/live"
    # IMAGE_FOLDER = "dataset_split/test/spoof"

    images = glob.glob(os.path.join(IMAGE_FOLDER, "*"))

    if len(images) == 0:
        raise FileNotFoundError(f"No images found inside {IMAGE_FOLDER}")

    image_path = random.choice(images)

print("\nTesting Image:")
print(image_path)

if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image not found:\n{image_path}")

# ==========================================
# Transform
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==========================================
# Read Image
# ==========================================

image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0).to(device)

# ==========================================
# Prediction
# ==========================================

classes = ["LIVE", "SPOOF"]

with torch.no_grad():
    output = model(image)
    probability = torch.softmax(output, dim=1)
    confidence, prediction = torch.max(probability, dim=1)

print("\n==============================")
print("Prediction :", classes[prediction.item()])
print(f"Confidence : {confidence.item()*100:.2f}%")
print("==============================")