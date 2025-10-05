"""Domain (runtime) models for detections and tracking"""

from pydantic import BaseModel


class BoundingBox(BaseModel):
    """Axis-aligned bounding box.

    Parameters
    ----------
    x_min : int
        Left (inclusive) pixel coordinate.
    y_min : int
        Top (inclusive) pixel coordinate.
    x_max : int
        Right (exclusive) pixel coordinate; must be greater than x_min.
    y_max : int
        Bottom (exclusive) pixel coordinate; must be greater than y_min.
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def as_xyxy(self) -> tuple[int, int, int, int]:
        """Return the bounding box as an (x_min, y_min, x_max, y_max) tuple."""

        return self.x_min, self.y_min, self.x_max, self.y_max


class VehicleTrack(BaseModel):
    """Tracked vehicle information.

    Parameters
    ----------
    object_id : int | None
        Tracking identifier (may be None if tracker does not assign one).
    class_name : str
        Name of the detected object class (e.g. car).
    bounding_box : BoundingBox
        Vehicle bounding box in the frame.
    """

    object_id: int | None
    class_name: str
    bounding_box: BoundingBox


class DetectedObject(BaseModel):
    """Detected vehicle + associated license plate (if detected).

    Parameters
    ----------
    object_id : int | None
        Tracking identifier propagated from the tracker.
    license_plate_text : str | None
        Extracted (OCR) license plate text or None if not recognised / no plate found.
    license_plate_confidence : float | None
        OCR confidence for the first recognised token (None if no plate or OCR skipped).
    predicted_object_type : str
        Detected vehicle class label.
    object_bounding_box : BoundingBox
        Vehicle bounding box.
    plate_bounding_box : BoundingBox | None
        License plate bounding box (region used for OCR) or None if no plate detected.
    """

    object_id: int | None
    license_plate_text: str | None
    license_plate_confidence: float | None
    predicted_object_type: str
    object_bounding_box: BoundingBox
    plate_bounding_box: BoundingBox | None


class VideoSource(BaseModel):
    """Represents an input video source (file path, RTSP URI, device index, etc.).

    Parameters
    ----------
    source : str
        Path or URI passed directly to cv2.VideoCapture.
    """

    source: str
