from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO


st.set_page_config(
    page_title="Pothole Detection | YOLOv8",
    page_icon="🛣️",
    layout="wide",
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best.pt"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights were not found at: {MODEL_PATH}"
        )
    return YOLO(str(MODEL_PATH))


st.title("🛣️ Pothole Detection using YOLOv8")
st.caption(
    "Upload a road image and the trained YOLOv8 model will identify visible potholes."
)

with st.expander("Model information"):
    st.markdown(
        """
        - **Model:** YOLOv8n
        - **Task:** Single-class pothole object detection
        - **Training epochs:** 30
        - **Validation images:** 133
        - **Validation instances:** 348
        - **Precision:** 0.846
        - **Recall:** 0.725
        - **mAP@50:** 0.8218
        - **mAP@50–95:** 0.5346
        """
    )

try:
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()

confidence = st.slider(
    "Minimum confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
)

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Upload a JPG or PNG road image to run pothole detection.")
else:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original image")
        st.image(image, use_container_width=True)

    with st.spinner("Running pothole detection..."):
        results = model.predict(
            source=np.array(image),
            conf=confidence,
            verbose=False,
        )

    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[..., ::-1]

    with col2:
        st.subheader("Detected potholes")
        st.image(annotated_rgb, use_container_width=True)

    boxes = result.boxes
    detection_count = 0 if boxes is None else len(boxes)

    st.subheader("Detection summary")

    m1, m2 = st.columns(2)
    m1.metric("Potholes detected", detection_count)

    if detection_count:
        confidences = boxes.conf.detach().cpu().numpy()
        m2.metric("Highest confidence", f"{confidences.max():.1%}")

        st.markdown("**Detection confidences**")
        for index, score in enumerate(confidences, start=1):
            st.write(f"Pothole {index}: {score:.1%}")
    else:
        m2.metric("Highest confidence", "—")
        st.info(
            "No potholes were detected above the selected confidence threshold. "
            "You can lower the threshold and try again."
        )

st.divider()
st.caption(
    "Academic computer-vision prototype. Detection quality depends on image quality, "
    "road conditions, and similarity to the training data."
)
