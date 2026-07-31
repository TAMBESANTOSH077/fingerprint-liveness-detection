import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from tqdm import tqdm

# =====================================================
# Device
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nDevice : {device}")

# =====================================================
# Dataset Path
# =====================================================

train_dir = "dataset_split/train"
val_dir = "dataset_split/val"

# =====================================================
# Image Transform
# =====================================================

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# =====================================================
# Dataset
# =====================================================

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

print("Classes :", train_dataset.classes)
print("Training Images :", len(train_dataset))
print("Validation Images :", len(val_dataset))

# =====================================================
# Model
# =====================================================

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
)

in_features = model.classifier[3].in_features

model.classifier[3] = nn.Linear(
    in_features,
    2
)

model = model.to(device)

print("\nModel Loaded Successfully")

# =====================================================
# Loss
# =====================================================

criterion = nn.CrossEntropyLoss()

# =====================================================
# Optimizer
# =====================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# =====================================================
# Training
# =====================================================

epochs = 10

best_accuracy = 0.0

os.makedirs("models", exist_ok=True)

for epoch in range(epochs):

    print(f"\nEpoch {epoch+1}/{epochs}")

    # ---------------- Training ----------------

    model.train()

    train_loss = 0
    train_correct = 0

    for images, labels in tqdm(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs,1)

        train_correct += (predicted==labels).sum().item()

    train_accuracy = train_correct / len(train_dataset)

    # ---------------- Validation ----------------

    model.eval()

    val_loss = 0
    val_correct = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs,1)

            val_correct += (predicted==labels).sum().item()

    val_accuracy = val_correct / len(val_dataset)

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Accuracy : {train_accuracy:.4f}")

    print(f"Validation Loss : {val_loss:.4f}")
    print(f"Validation Accuracy : {val_accuracy:.4f}")

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "models/liveness_model.pth"
        )

        print("Best Model Saved")

print("\n===================================")
print("Training Completed")
print("Best Validation Accuracy :", best_accuracy)
print("Model Saved : models/liveness_model.pth")
print("===================================")