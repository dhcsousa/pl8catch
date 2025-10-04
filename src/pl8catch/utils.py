"""Backend streaming utilities (combined multipart frames + detections)."""

import json
from typing import AsyncGenerator, Tuple
import asyncio
import cv2
import pytesseract
import numpy as np
from loguru import logger
from ultralytics import YOLO

from pl8catch.data_model import AppConfig, DetectedObject


def _detect_plate(
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
    - `object_bounding_box`: A tuple representing the bounding box of the detected object (x_min, y_min, x_max, y_max).
    - `plate_bounding_box`: A tuple representing the bounding boxes of detected license plate within the object (x_min, y_min, x_max, y_max).
    """

    logger.debug("Starting object tracking on frame of shape {}", image.shape)
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
                object_bounding_box = (
                    x_min_vehicle,
                    y_min_vehicle,
                    x_max_vehicle,
                    y_max_vehicle,
                )  # Store coordinates as a tuple

            logger.debug("Detected object id={}, class={}, bbox=%s", object_id, class_name, object_bounding_box)
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

                    plate_bounding_box = (x_min_plate, y_min_plate, x_max_plate, y_max_plate)
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
                    logger.debug(
                        "Plate candidate for object_id=%s plate_bbox=%s text=%s conf=%s",
                        object_id,
                        plate_bounding_box,
                        license_plate_text,
                        license_plate_confidence,
                    )
                    detected_objects.append(
                        DetectedObject(
                            object_id=object_id,
                            license_plate_text=license_plate_text,
                            license_plate_confidence=license_plate_confidence,
                            predicted_object_type=class_name,
                            object_bounding_box=object_bounding_box,
                            plate_bounding_box=plate_bounding_box,
                        ),
                    )

    logger.debug("Frame processing complete: %d objects detected", len(detected_objects))
    return detected_objects


def _plot_objects_in_image(
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

    logger.trace("Annotating %d objects on frame", len(detected_objects))
    image_with_annotations = image.copy()

    for detected_object in detected_objects:
        object_bb = detected_object.object_bounding_box
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
        detected_plate = detected_object.plate_bounding_box
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


def _process_frame(
    frame: np.ndarray,
    frame_index: int,
    yolo_object_model: YOLO,
    yolo_plate_model: YOLO,
    config: AppConfig,
) -> Tuple[list[DetectedObject], bytes, dict]:
    """Synchronous inference + annotation -> returns detections, jpeg bytes, and metadata payload."""
    detected_objects = _detect_plate(frame, yolo_object_model, yolo_plate_model, config)
    annotated = _plot_objects_in_image(frame, detected_objects)
    ok, jpeg_mat = cv2.imencode(".jpg", annotated)
    if not ok:
        raise RuntimeError("JPEG encode failed")
    jpeg_bytes = jpeg_mat.tobytes()
    detections_dicts = [o.model_dump() for o in detected_objects]
    payload = {"frame_index": frame_index, "detections": detections_dicts}
    return detected_objects, jpeg_bytes, payload


async def stream_frame_and_detections_multipart(
    video: cv2.VideoCapture, yolo_object_model: YOLO, yolo_plate_model: YOLO, config: AppConfig
) -> AsyncGenerator[bytes, None]:
    """Yield alternating JSON and JPEG parts over multipart/mixed without duplicate inference.

    Pattern per frame:
        --frame\r\n
        Content-Type: application/json\r\n\r\n
        { ...metadata... }\r\n
        --frame\r\n
        Content-Type: image/jpeg\r\nContent-Length: N\r\n\r\n
        <jpeg bytes>\r\n
    Terminates with: --frame--\r\n
    """
    boundary = b"--frame"
    frame_index = 0
    while True:
        ok, frame = video.read()
        if not ok:
            logger.info("Multipart stream ended at frame_index=%d", frame_index)
            break
        frame_index += 1
        try:
            # Run blocking processing in a thread to avoid blocking the event loop
            _det, jpeg_bytes, payload = await asyncio.to_thread(
                _process_frame, frame, frame_index, yolo_object_model, yolo_plate_model, config
            )
            json_part = (
                boundary
                + b"\r\nContent-Type: application/json\r\n\r\n"
                + json.dumps(payload, separators=(",", ":")).encode()
                + b"\r\n"
            )
            yield json_part
            jpeg_part = (
                boundary
                + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg_bytes)).encode()
                + b"\r\n\r\n"
                + jpeg_bytes
                + b"\r\n"
            )
            yield jpeg_part
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error processing frame_index=%d: %s", frame_index, exc)
            continue
    yield b"--frame--\r\n"
