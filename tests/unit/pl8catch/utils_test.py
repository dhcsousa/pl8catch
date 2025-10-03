import numpy as np
import pytest
from pl8catch.data_model import DetectedObject
from pl8catch.utils import detect_plate, plot_objects_in_image


def test_detect_plate(config, models, test_image):
    # Mock YOLO models and config
    yolo_object_model, yolo_plate_model = models

    # Call the function with sample data
    detected_objects = detect_plate(test_image, yolo_object_model, yolo_plate_model, config)

    # Assert that the function returns a list
    assert len(detected_objects) == 1
    detected_car = detected_objects[0]
    assert detected_car.object_id == 1
    assert isinstance(detected_car.license_plate_text, str)
    assert isinstance(detected_car.license_plate_confidence, float)
    assert isinstance(detected_car.predicted_object_type, str)
    assert isinstance(detected_car.object_bounding_box, tuple)
    assert len(detected_car.object_bounding_box) == 4
    assert isinstance(detected_car.plate_bounding_box, tuple)
    assert len(detected_car.plate_bounding_box) == 4


@pytest.fixture()
def sample_detected_objects():
    # Sample DetectedObject objects for testing
    detected_objects = [
        DetectedObject(
            object_id=1,
            license_plate_text="ABC123",
            license_plate_confidence=0.9,
            predicted_object_type="car",
            object_bounding_box=(50, 50, 150, 150),
            plate_bounding_box=(70, 70, 120, 90),
        ),
        DetectedObject(
            object_id=2,
            license_plate_text="XYZ456",
            license_plate_confidence=0.8,
            predicted_object_type="truck",
            object_bounding_box=(200, 200, 300, 300),
            plate_bounding_box=(220, 220, 280, 240),
        ),
    ]
    return detected_objects


def test_plot_objects_in_image(sample_detected_objects):
    # Create a sample image
    image = np.zeros((800, 800, 3), dtype=np.uint8)

    # Call the function with sample data
    annotated_image = plot_objects_in_image(image, sample_detected_objects)

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


def test_stream_detected_frames():
    # stream_detected_frames()
    pass
