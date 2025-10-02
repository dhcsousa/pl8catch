"""Basic utils for the backend of pl8catch"""

import base64
import json
from typing import AsyncGenerator
import cv2
import pytesseract
import numpy as np
from ultralytics import YOLO

from pl8catch.data_model import AppConfig, DetectedObject


def detect_plate(
    image: np.ndarray, yolo_object_model: YOLO, yolo_plate_model: YOLO, config: AppConfig
) -> list[DetectedObject]:
    """
    This function takes an image, two YOLO models (one for general object detection, another for license plate detection),
    and a configuration dictionary as input and returns a list of `DetectedObject` objects.

    Parameters
    ----------
    image : numpy.ndarray
        A NumPy array representing the image to be processed.
    yolo_object_model : YOLO model instance
        A YOLO model instance used for general object detection (e.g., cars, motorcycles).
    yolo_plate_model : YOLO model instance
        A YOLO model instance trained specifically for license plate detection.
    config : AppConfig
        YAMLConfig containing configuration parameters for various steps (e.g., license plate OCR resizing threshold).

    Returns
    ----------
    list of DetectedObject objects
        A list of `DetectedObject` objects. Each object contains:
        - `license_plate_text`: A list of recognized characters from the license plate (might be empty if not detected).
        - `license_plate_confidence`: A list of confidence scores for each recognized character (might be empty if not detected).
        - `predicted_object_type`: The type of object predicted by the general YOLO model (e.g., "car").
        - `object_biding_box`: A tuple representing the bounding box of the detected object (x_min, y_min, x_max, y_max).
        - `plate_biding_box`: A tuple representing the bounding boxes of detected license plate within the object (x_min, y_min, x_max, y_max).
    """

    yolo_predictions = yolo_object_model.track(
        image, persist=True, classes=[2, 3, 5, 7], verbose=False, tracker="bytetrack.yaml"
    )  # car, motorcycle, bus, truck
    detected_objects = []

    for result in yolo_predictions:
        if result.boxes:
            for box in result.boxes:
                if box.id is not None:
                    object_id = int(box.id.item())
                else:
                    object_id = None
                x_min_vehicle, y_min_vehicle, x_max_vehicle, y_max_vehicle = (
                    int(box.xyxy[0][0]),
                    int(box.xyxy[0][1]),
                    int(box.xyxy[0][2]),
                    int(box.xyxy[0][3]),
                )
                class_name = f"{result.names[int(box.cls[0])]}"
                object_biding_box = (
                    x_min_vehicle,
                    y_min_vehicle,
                    x_max_vehicle,
                    y_max_vehicle,
                )  # Store coordinates as a tuple

            plates = yolo_plate_model.predict(
                image[y_min_vehicle:y_max_vehicle, x_min_vehicle:x_max_vehicle], verbose=False, max_det=1
            )  # Predict plates only within the car box
            for plate in plates:
                for box in plate.boxes:
                    # TODO (@Daniel.Sousa): Only the plate with highest confidence to be plotted?
                    x_min_relative_plate, y_min_relative_plate, x_max_relative_plate, y_max_relative_plate = (
                        int(box.xyxy[0][0]),
                        int(box.xyxy[0][1]),
                        int(box.xyxy[0][2]),
                        int(box.xyxy[0][3]),
                    )
                    x_min_plate = x_min_relative_plate + x_min_vehicle
                    x_max_plate = x_max_relative_plate + x_min_vehicle
                    y_min_plate = y_min_relative_plate + y_min_vehicle
                    y_max_plate = y_max_relative_plate + y_min_vehicle

                    plate_biding_box = (x_min_plate, y_min_plate, x_max_plate, y_max_plate)
                    # TODO: Possible improvement, only append to detected_plate, after text is detected?

                    # OCR the license plate
                    original_roi = image[y_min_plate:y_max_plate, x_min_plate:x_max_plate]

                    # I noticed that if the resolution/area of the license plate is small results degrade, increasing it improves them
                    treated_roi = original_roi.copy()
                    while treated_roi.shape[0] * treated_roi.shape[1] < config.license_plate_ocr.resizing_threshold:
                        treated_roi = cv2.resize(treated_roi, (2 * treated_roi.shape[1], 2 * treated_roi.shape[0]))

                    # Normalizing and thresholding image
                    norm_img = np.zeros((treated_roi.shape[0], treated_roi.shape[1]))
                    treated_roi = cv2.normalize(treated_roi, norm_img, 0, 255, cv2.NORM_MINMAX)
                    treated_roi = cv2.threshold(treated_roi, 127, 255, cv2.THRESH_BINARY)[1]
                    treated_roi = cv2.GaussianBlur(treated_roi, (1, 1), 0)

                    # Extract Data from License Plate Image Segment using pytesseract
                    pytesseract_data = pytesseract.image_to_data(
                        treated_roi, config=config.license_plate_ocr.pytesseract_config, output_type="data.frame"
                    )
                    pytesseract_data = pytesseract_data[pytesseract_data["conf"] > 0.1][["conf", "text"]]

                    # Extract license plate text and confidence
                    license_plate_text, license_plate_confidence = (
                        (str(pytesseract_data.iloc[0]["text"]), pytesseract_data.iloc[0]["conf"])
                        if not pytesseract_data.empty
                        else (None, None)
                    )

                    # Append the recognized character to the list
                    detected_objects.append(
                        DetectedObject(
                            object_id=object_id,
                            license_plate_text=license_plate_text,
                            license_plate_confidence=license_plate_confidence,
                            predicted_object_type=class_name,
                            object_biding_box=object_biding_box,
                            plate_biding_box=plate_biding_box,
                        ),
                    )

    return detected_objects


def plot_objects_in_image(
    image: np.ndarray, detected_objects: list[DetectedObject], line_thickness: int = 5, font_scale: int = 2
) -> np.ndarray:
    """
    Plot detected objects in the input image with annotations.

    Parameters
    ----------
    image : numpy.ndarray
        A NumPy array representing the image to be plotted.
    detected_objects : list of DetectedObject objects
        A list of DetectedObject objects containing information about detected objects and their bounding boxes.
    line_thickness : int
        Line thickness for the binding boxes rectangles
    font_scale: int
        Parameter used to control the text around the binding boxes

    Returns
    ----------
    numpy.ndarray
        An annotated image with detected objects plotted. Bounding boxes and object types are annotated.

    Notes
    ----------
    This function modifies a copy of the input image to add annotations for each detected object and its corresponding license plate(s).
    Bounding boxes are drawn around each detected object and license plate(s), and the predicted object type is annotated above each object bounding box.
    """

    image_with_annotations = image.copy()

    for detected_object in detected_objects:
        object_bb = detected_object.object_biding_box
        cv2.rectangle(
            image_with_annotations,
            (object_bb[0], object_bb[1]),
            (object_bb[2], object_bb[3]),
            (255, 0, 0),
            line_thickness,
        )
        cv2.putText(
            image_with_annotations,
            f"ID: {detected_object.object_id}, TYPE: {detected_object.predicted_object_type}",
            (object_bb[0], object_bb[1] - 10),
            cv2.FONT_HERSHEY_PLAIN,
            font_scale,
            (255, 0, 0),
            font_scale,
        )
        detected_plate = detected_object.plate_biding_box
        cv2.rectangle(
            image_with_annotations,
            (detected_plate[0], detected_plate[1]),
            (detected_plate[2], detected_plate[3]),
            (255, 0, 0),
            line_thickness,
        )
        cv2.putText(
            image_with_annotations,
            "plate",
            (detected_plate[0], detected_plate[1] - 10),
            cv2.FONT_HERSHEY_PLAIN,
            font_scale,
            (255, 0, 0),
            font_scale,
        )

    return image_with_annotations


async def stream_detected_frames(
    video: cv2.VideoCapture, yolo_object_model: YOLO, yolo_plate_model: YOLO, config: AppConfig
) -> AsyncGenerator:  # TODO (@Daniel.Sousa): Unit tests
    """
    Stream video frames with object and license plate detection.

    Parameters
    ----------
    video : cv2.VideoCapture
        A cv2.VideoCapture object to capture video frames.
    yolo_object_model : YOLO model instance
        A YOLO model instance used for general object detection (e.g., cars, motorcycles).
    yolo_plate_model : YOLO model instance
        A YOLO model instance trained specifically for license plate detection.
    config : AppConfig
        YAMLConfig containing configuration parameters for various steps (e.g., license plate OCR resizing threshold).

    Returns
    ----------
    AsyncGenerator
        A generator yielding video frames as JPEG-encoded bytes and detection data as json.

    Notes
    ----------
    This function reads video frames from the provided video capture object,
    detects objects and license plates in each frame, annotates the frame,
    and then encodes it as JPEG. The resulting bytes are streamed in a format
    compatible with MJPEG (multipart/x-mixed-replace).
    """
    while True:
        success, frame = video.read()
        if not success:
            break

        # Detect objects
        detected_objects = detect_plate(frame, yolo_object_model, yolo_plate_model, config)
        annotated_image = plot_objects_in_image(frame, detected_objects)

        # Encode frame as JPEG
        _, jpeg = cv2.imencode(".jpg", annotated_image)

        # Convert frame to base64 string
        frame_base64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")

        # Convert detected objects to dictionaries
        detected_objects_dicts = [obj.model_dump() for obj in detected_objects]

        # Create the payload
        payload = {"frame": frame_base64, "detections": detected_objects_dicts}

        # Yield the payload as JSON
        yield (b"data: " + json.dumps(payload).encode() + b"\n\n")
