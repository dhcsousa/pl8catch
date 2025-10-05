from pl8catch.core.domain_models import BoundingBox, DetectedObject


def test_bounding_box_tuple():
    bb = BoundingBox(x_min=1, y_min=2, x_max=10, y_max=20)
    assert bb.as_xyxy() == (1, 2, 10, 20)


def test_detected_object_creation():
    bb_vehicle = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50)
    bb_plate = BoundingBox(x_min=10, y_min=5, x_max=60, y_max=25)
    obj = DetectedObject(
        object_id=123,
        license_plate_text="ABC123",
        license_plate_confidence=0.85,
        predicted_object_type="car",
        object_bounding_box=bb_vehicle,
        plate_bounding_box=bb_plate,
    )
    assert obj.object_id == 123
    assert obj.license_plate_text == "ABC123"
    assert obj.license_plate_confidence == 0.85
    assert obj.object_bounding_box.as_xyxy() == (0, 0, 100, 50)
    assert obj.plate_bounding_box.as_xyxy() == (10, 5, 60, 25)
