import streamlit as st
from detect import detect
import os
import cv2
import tempfile

st.title("Military Soldier Safety & Weapon Detection")

st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload Image or Video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi']
)
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
project_dir = st.sidebar.text_input(
    "Weights Project Path", "runs/train/military_yolo"
)

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.close()
    source_path = tfile.name

    st.sidebar.write("Processing...")
    # Run detection and get output path
    try:
        annotated_path = detect(
            source=source_path,
            weights=os.path.join(project_dir, 'weights', 'best.pt'),
            conf_thres=confidence,
            project_dir=project_dir
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Display results
    if annotated_path and os.path.exists(annotated_path):
        if uploaded_file.type.startswith('image'):
            img = cv2.imread(annotated_path)
            st.image(img, channels="BGR", caption="Annotated Image")
        else:
            st.video(annotated_path)
    else:
        st.error("Detection failed or no output generated.")