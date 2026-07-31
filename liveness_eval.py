import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score
)

# ==========================================================
# Device
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device :", device)

# ==========================================================
# Test Dataset
# ==========================================================

test_dir = "dataset_split/test"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

print("Testing Images :", len(test_dataset))

# ==========================================================
# Model
# ==========================================================

model = models.mobilenet_v3_small(weights=None)

in_features = model.classifier[3].in_features

model.classifier[3] = nn.Linear(in_features,2)

model.load_state_dict(
    torch.load(
        "models/liveness_model.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

print("Model Loaded Successfully")

# ==========================================================
# Evaluation
# ==========================================================

true_labels = []
predicted_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs,1)

        true_labels.extend(labels.numpy())

        predicted_labels.extend(predictions.cpu().numpy())

# ==========================================================
# Accuracy
# ==========================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

print("\nAccuracy :", round(accuracy*100,2),"%")

# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report\n")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=test_dataset.classes
    )
)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_dataset.classes
)

disp.plot(cmap="Blues")

os.makedirs("outputs",exist_ok=True)

plt.savefig("outputs/confusion_matrix.png")

plt.show()

print("\nConfusion Matrix Saved")

print("\nEvaluation Completed")