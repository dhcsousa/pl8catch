import cv2
import pytest
import numpy as np
import requests
from pathlib import Path
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

    # Provide a lightweight fallback for the license plate model when the weights file
    # is not available in CI or local test environments. This avoids test failures due
    # to missing artifact (e.g., yolo_runs/run_1/weights/best.pt) while still exercising
    # the detection pipeline downstream (detect_plate and plotting logic).
    license_plate_weights = Path(config.models.license_plate)

    if license_plate_weights.exists():
        yolo_plate_model = YOLO(config.models.license_plate)
    else:

        class _DummyBox:
            """Mimics a single YOLO box structure with xyxy coordinates."""

            def __init__(self):
                # (x_min, y_min, x_max, y_max) small dummy plate region
                self.xyxy = np.array([[0, 0, 50, 30]])

        class _DummyResult:
            def __init__(self):
                self.boxes = [_DummyBox()]

        class DummyPlateModel:
            """Minimal stub replicating the predict() API used in tests.

            Returns a list with a single result object containing one bounding box.
            """

            def predict(self, *args, **kwargs):
                """Return a list with a single dummy result containing one bounding box."""
                return [_DummyResult()]

        yolo_plate_model = DummyPlateModel()  # type: ignore[assignment]

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
