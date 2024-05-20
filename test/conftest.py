import cv2
import pytest
import numpy as np
import requests
from ultralytics import YOLO

from pl8catch.data_model import CONFIG, YAMLConfig


@pytest.fixture()
def config() -> YAMLConfig:
    return CONFIG


@pytest.fixture()
def models() -> tuple[YOLO, YOLO]:
    yolo_object_model = YOLO(CONFIG.models.object_detection)
    yolo_plate_model = YOLO(CONFIG.models.license_plate)

    return yolo_object_model, yolo_plate_model


@pytest.fixture()
def test_image() -> np.ndarray:
    # Image URL
    # Image obtained from https://www.roadandtrack.com/new-cars/
    image_address = "https://hips.hearstapps.com/hmg-prod/images/rs357223-1626456047.jpg"

    # Download the image
    response = requests.get(image_address, timeout=10)
    response.raise_for_status()  # Raise an exception if the download fails

    # Read the image using OpenCV
    image_np_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)

    # Return the image
    return image
