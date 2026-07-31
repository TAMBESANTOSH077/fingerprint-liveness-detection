import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

# -------------------------------
# Classes
# -------------------------------
classes = ["live", "spoof"]

# -------------------------------
# Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Load Model
# -------------------------------
model = models.mobilenet_v3_small(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    2
)

model.load_state_dict(
    torch.load(
        "models/liveness_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("Model Loaded Successfully")

# -------------------------------
# Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam.")
    exit()

print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(rgb)

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image)

        prob = torch.softmax(output, dim=1)

        conf, pred = torch.max(prob, 1)

    label = classes[pred.item()].upper()
    confidence = conf.item() * 100

    if label == "LIVE":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    cv2.putText(
        frame,
        f"{label} ({confidence:.2f}%)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Fingerprint Liveness Detection", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()