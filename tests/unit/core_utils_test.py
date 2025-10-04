import numpy as np
import pandas as pd
import pytesseract
from loguru import logger
import pytest

from pl8catch.config.app_config import AppConfig
from pl8catch.core.model import BoundingBox, DetectedObject
from pl8catch.core.utils import (
    _detect_plate,
    _ocr_plate,
    _plot_objects_in_image,
    _prepare_plate_roi,
    _process_frame,
)
from pl8catch.logging.setup import configure_logging


class DummyTrackBox:
    def __init__(self):
        import numpy as _np

        self.id = _np.array([1])
        self.cls = _np.array([2])
        self.xyxy = _np.array([[0, 0, 10, 10]])


class DummyTrackResult:
    def __init__(self):
        self.boxes = [DummyTrackBox()]
        self.names = {2: "car"}


class DummyVehicleModel:
    def track(self, *a, **k):
        return [DummyTrackResult()]


class DummyPlateBox:
    def __init__(self):
        import numpy as _np

        self.xyxy = _np.array([[1, 1, 5, 4]])


class DummyPlateResult:
    def __init__(self):
        self.boxes = [DummyPlateBox()]


class DummyPlateModel:
    def predict(self, *a, **k):
        return [DummyPlateResult()]


def test_ocr_plate_returns_first(monkeypatch):
    data = pd.DataFrame({"conf": [0.2, 0.5], "text": ["ABC123", "DEF456"]})

    def fake_image_to_data(*a, **k):
        return data

    monkeypatch.setattr(pytesseract, "image_to_data", fake_image_to_data)
    text, conf = _ocr_plate(np.zeros((10, 10), dtype=np.uint8), "--psm 7")
    assert text == "ABC123" and conf == 0.2


def test_configure_logging_changes_level():
    configure_logging("INFO")
    # Log something and ensure no exception; loguru stores handlers internally
    logger.info("Test info message")
    # Check that at least one handler exists after configuration
    assert logger._core.handlers


def test_prepare_plate_roi_preserves_binary():
    img = (np.random.rand(10, 10) * 255).astype("uint8")
    out = _prepare_plate_roi(img, min_area=50)  # area already >=50 so minimal processing
    # Ensure output has same or larger area
    assert out.shape[0] * out.shape[1] >= 100  # may upscale depending on random dims
    # Values limited to 0..255
    assert out.min() >= 0 and out.max() <= 255


def test_process_frame_single_detection(monkeypatch):
    frame = (np.zeros((20, 20, 3))).astype("uint8")
    vehicle_model = DummyVehicleModel()
    plate_model = DummyPlateModel()

    # Patch internal OCR functions to avoid tesseract
    from pl8catch.core import utils as utils_mod

    def fake_prepare_roi(roi, min_area):
        return roi

    def fake_ocr(img, cfg):
        return "ABC123", 0.99

    monkeypatch.setattr(utils_mod, "_prepare_plate_roi", fake_prepare_roi)
    monkeypatch.setattr(utils_mod, "_ocr_plate", fake_ocr)

    cfg = AppConfig(
        server={"host": "127.0.0.1", "port": 8000},
        models={"object_detection": "obj.pt", "license_plate": "lp.pt"},
        license_plate_ocr={"resizing_threshold": 1, "pytesseract_config": "--psm 7"},
    )

    det, jpeg_bytes, payload = _process_frame(frame, 1, vehicle_model, plate_model, cfg)
    assert len(det) == 1
    d = det[0]
    assert d.license_plate_text == "ABC123"
    assert payload["frame_index"] == 1
    assert len(payload["detections"]) == 1


def test_detect_plate(config, models, test_image):
    # Mock YOLO models and config
    yolo_object_model, yolo_plate_model = models

    # Call the function with sample data
    detected_objects = _detect_plate(test_image, yolo_object_model, yolo_plate_model, config)

    # Assert that the function returns a list
    assert len(detected_objects) == 1
    detected_car = detected_objects[0]
    assert detected_car.object_id == 1
    assert isinstance(detected_car.license_plate_text, str)
    assert isinstance(detected_car.license_plate_confidence, float)
    assert isinstance(detected_car.predicted_object_type, str)
    assert isinstance(detected_car.object_bounding_box, BoundingBox)
    assert isinstance(detected_car.plate_bounding_box, BoundingBox)


@pytest.fixture()
def sample_detected_objects():
    # Sample DetectedObject objects for testing
    detected_objects = [
        DetectedObject(
            object_id=1,
            license_plate_text="ABC123",
            license_plate_confidence=0.9,
            predicted_object_type="car",
            object_bounding_box=BoundingBox(x_min=50, y_min=50, x_max=150, y_max=150),
            plate_bounding_box=BoundingBox(x_min=70, y_min=70, x_max=120, y_max=90),
        ),
        DetectedObject(
            object_id=2,
            license_plate_text="XYZ456",
            license_plate_confidence=0.8,
            predicted_object_type="truck",
            object_bounding_box=BoundingBox(x_min=200, y_min=200, x_max=300, y_max=300),
            plate_bounding_box=BoundingBox(x_min=220, y_min=220, x_max=280, y_max=240),
        ),
    ]
    return detected_objects


def test_plot_objects_in_image(sample_detected_objects):
    # Create a sample image
    image = np.zeros((800, 800, 3), dtype=np.uint8)

    # Call the function with sample data
    annotated_image = _plot_objects_in_image(image, sample_detected_objects)

    # Check if the annotated image has the correct shape
    assert annotated_image.shape == image.shape

    # Check if at least one pixel has the blue color (annotation)
    blue_pixel_found = False
    for row in annotated_image:
        for pixel in row:
            if (pixel == [255, 0, 0]).all():  # Blue color represented as [255, 0, 0] in OpenCV
                blue_pixel_found = True
                break
        if blue_pixel_found:
            break
    assert blue_pixel_found  # "At least one pixel with blue color should be present"
