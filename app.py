import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
from model import ResNet50

st.set_page_config(page_title="ResNet50 Image Classifier (PyTorch)", layout="centered")

st.title("🧠 ResNet50 Image Classifier")
st.write("Upload an image — the ResNet50 model will classify it")

@st.cache_resource
def load_model(weights_file, device="cpu"):
    # model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = ResNet50()
    model.eval()  # evaluation mode
    model = model.to(device)
    checkpoint = torch.load(weights_file, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return model

weights_file = "run5-epoch89.pth"
model = load_model(weights_file)

# Get class labels
imagenet_classes = models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Imagenet normalization
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# File Upload
# -----------------------------
col1, col2 = st.columns([2, 1])
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file).convert("RGB")

    # Two-column layout: image (left), prediction (right)
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image for model
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Run inference (CPU)
    with st.spinner("Classifying..."):
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get top class
        top_prob, top_class = torch.topk(probabilities, 1)
        label = imagenet_classes[top_class.item()]
        confidence = top_prob.item() * 100

    # Display results: Show the image on left, and label info on right
    with col2:
        st.subheader("Prediction")
        st.success(f"**Label:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}%")


    st.markdown("---")
    st.caption("Model: ResNet50 trained on ImageNet1K (CPU inference)")


    # # -----------------------------
    # # Bounding Box (simple visual cue)
    # # -----------------------------
    # open_cv_image = np.array(img)
    # open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    #
    # gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # if contours:
    #     c = max(contours, key=cv2.contourArea)
    #     x, y, w, h = cv2.boundingRect(c)
    #     boxed_img = open_cv_image.copy()
    #     cv2.rectangle(boxed_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    #     st.image(cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB), caption="Bounding Box Approximation")
    #     st.write(f"**Bounding Box Coordinates:** (x={x}, y={y}, w={w}, h={h})")
    # else:
    #     st.warning("No clear object found to draw bounding box.")
