from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
from torchvision import models, transforms

app = FastAPI(title="Fingerprint Liveness Detection API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["LIVE", "SPOOF"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features,2)

model.load_state_dict(
    torch.load(
        "models/liveness_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()


@app.get("/")
def home():
    return {"message":"Fingerprint Liveness Detection API"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(BytesIO(await file.read())).convert("RGB")

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probability = torch.softmax(output,1)
        confidence, prediction = torch.max(probability,1)

    return {
        "prediction": classes[prediction.item()],
        "confidence": round(confidence.item()*100,2)
    }