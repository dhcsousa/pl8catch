import yaml
import cv2
from ultralytics import YOLO

import streamlit as st

from pl8catch.utils import detect_plate, plot_objects_in_image

yolo_model = YOLO("models/yolov9c.pt")
yolo_plate_model = YOLO("models/license_plate_yolov9c.pt")

with open("config.yaml") as stream:
    config = yaml.safe_load(stream)


video_address = "demo_files/demo_video.mp4"

video = cv2.VideoCapture(video_address)

# Setting page layout
st.set_page_config(page_title="License Plate Detection using YOLOv9", layout="wide", initial_sidebar_state="expanded")

# Main page heading
st.title("License Plate Detection using YOLOv9")

st_frame = st.empty()

while True:
    check, frame = video.read()
    if not check:
        print("Error: Unable to read frame. Video might have ended.")
        break

    detected_objects = detect_plate(frame, yolo_model, yolo_plate_model, config)
    annotated_image = plot_objects_in_image(frame, detected_objects)

    # Limiting the height of the displayed image to 500 pixels
    height, width, _ = annotated_image.shape
    max_height = 500
    if height > max_height:
        scale = max_height / height
        width = int(width * scale)
        height = max_height
        annotated_image = cv2.resize(annotated_image, (width, height))

    st_frame.image(annotated_image, caption="Detected Video", channels="BGR")

    # TODO: car detection

video.release()
