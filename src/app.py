from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

from paths import MODEL_PATH, require_file
from visualization import draw_boxes


@st.cache_resource
def load_model(model_path):
    return YOLO(str(require_file(model_path)))


def main():
    st.set_page_config(page_title="Shelf Detector", layout="wide")
    st.title("Shelf Detection Dashboard")
    model_path = st.text_input("Checkpoint path", str(MODEL_PATH))
    uploaded_files = st.file_uploader(
        "Upload shelf images", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    confidence = st.slider("Confidence threshold", 0.0, 1.0, 0.35, 0.05)
    nms_iou = st.slider("NMS IoU threshold", 0.0, 1.0, 0.50, 0.05)
    thickness = st.slider("Box border thickness", 1, 8, 4, 1)
    if not uploaded_files:
        return
    try:
        model = load_model(str(Path(model_path).expanduser().resolve()))
    except (FileNotFoundError, OSError, RuntimeError) as error:
        st.error(f"Cannot load checkpoint: {error}")
        return
    for uploaded_file in uploaded_files:
        try:
            with Image.open(uploaded_file) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            st.error(f"Cannot open {uploaded_file.name}: {error}")
            continue
        result = model.predict(
            source=image, conf=confidence, iou=nms_iou, imgsz=640,
            max_det=718, verbose=False,
        )[0]
        boxes = result.boxes.xyxy.cpu().tolist()
        plotted = draw_boxes(image, boxes, thickness)
        st.subheader(uploaded_file.name)
        original_column, prediction_column = st.columns(2)
        original_column.image(image, caption="Original", width="stretch")
        prediction_column.image(plotted, caption="Detections", width="stretch")
        st.metric("Detected objects", len(boxes))


if __name__ == "__main__":
    main()
