import cv2
import pytest
import numpy as np
import requests
from ultralytics import YOLO

from pl8catch.data_model import AppConfig


@pytest.fixture(scope="session")
def config() -> AppConfig:
    """Load application configuration once per test session."""
    return AppConfig.from_file("configs/backend.yaml")


@pytest.fixture(scope="session")
def models(config: AppConfig) -> tuple[YOLO, YOLO]:
    """Instantiate YOLO models only once; they are relatively heavy objects."""
    yolo_object_model = YOLO(config.models.object_detection)
    yolo_plate_model = YOLO(config.models.license_plate)
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
