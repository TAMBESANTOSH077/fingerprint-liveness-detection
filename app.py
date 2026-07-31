import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(
    page_title="Fingerprint Liveness Detection",
    page_icon="🛡️",
    layout="centered"
)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Classes
# -----------------------------
classes = ["LIVE", "SPOOF"]

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

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

    return model

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🛡️ Fingerprint Liveness Detection")

st.write("Upload a fingerprint image.")

uploaded_file = st.file_uploader(
    "Choose Fingerprint Image",
    type=["png","jpg","jpeg","bmp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Fingerprint",
        use_container_width=True
    )

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(img)

        prob = torch.softmax(output,1)

        conf, pred = torch.max(prob,1)

    label = classes[pred.item()]
    confidence = conf.item()*100

    st.divider()

    if label == "LIVE":

        st.success(f"Prediction : {label}")

    else:

        st.error(f"Prediction : {label}")

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )