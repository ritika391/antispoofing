import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import plotly.graph_objects as go

from src.model import load_model
from src.config import MODEL_CONFIG, train_cfg, model_cfg
from utils.augmentation import get_val_transforms

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="FaceGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def get_model():

    model = load_model(
        MODEL_CONFIG["best_model"],
        train_cfg.device
    )

    return model

model = get_model()

device = torch.device(train_cfg.device)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("🛡️ FaceGuard AI")
st.subheader("Face Anti-Spoofing Detection System")

st.markdown("---")

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

col1,col2,col3 = st.columns(3)

with col1:
    st.metric("Accuracy","97.5%")

with col2:
    st.metric("AUC","0.975")

with col3:
    st.metric("Epochs","5")

st.markdown("---")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Face Image",
    type=["jpg","jpeg","png"]
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if uploaded_file:
    
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=350
    )

    if st.button("Predict"):

        img = np.array(image)

        transform = get_val_transforms(
            model_cfg.image_size
        )

        transformed = transform(image=img)

        tensor = transformed["image"].unsqueeze(0)

        tensor = tensor.to(device)

        with torch.no_grad():

            logits,_ = model(tensor)

            probs = torch.softmax(
                logits,
                dim=1
            )[0]

        real_prob = probs[0].item()
        spoof_prob = probs[1].item()

        pred = np.argmax(
            [real_prob,spoof_prob]
        )

        if pred == 0:
            label = "✅ REAL FACE"
            confidence = real_prob
            st.success(label)

        else:
            label = "🚨 SPOOF FACE"
            confidence = spoof_prob
            st.error(label)

        st.write(
            f"Confidence : {confidence*100:.2f}%"
        )

        # ------------------------------------------
        # CHART
        # ------------------------------------------

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=["Real","Spoof"],
                y=[
                    real_prob*100,
                    spoof_prob*100
                ]
            )
        )

        fig.update_layout(
            title="Prediction Probabilities"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

st.markdown("---")

st.info(
    "FaceGuard MobileNetV2 Anti-Spoofing System"
)