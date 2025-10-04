"""Backend streaming utilities (combined multipart frames + detections)."""

import asyncio
import json
from typing import AsyncGenerator, Iterator, Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import TesseractError  # specific error raised by OCR backend
from loguru import logger
from ultralytics import YOLO

from pl8catch.config import AppConfig
from pl8catch.core.model import BoundingBox, DetectedObject, VehicleTrack


def _track_vehicles(image: np.ndarray, model: YOLO) -> Iterator[VehicleTrack]:
    """Track vehicles in a frame.

    Parameters
    ----------
    image : numpy.ndarray
        Input BGR frame.
    model : ultralytics.YOLO
        YOLO model instance configured for multi-class vehicle detection with tracking.

    Yields
    ------
    VehicleTrack
        A vehicle track object containing (optional) ID, class name and bounding box.
    """
    predictions = model.track(image, persist=True, classes=[2, 3, 5, 7], verbose=False, tracker="bytetrack.yaml")
    for result in predictions:
        if not result.boxes:
            continue
        for box in result.boxes:
            object_id = int(box.id.item()) if box.id is not None else None
            x_min, y_min, x_max, y_max = (
                int(box.xyxy[0][0]),
                int(box.xyxy[0][1]),
                int(box.xyxy[0][2]),
                int(box.xyxy[0][3]),
            )
            class_name = f"{result.names[int(box.cls[0])]}"
            yield VehicleTrack(
                object_id=object_id,
                class_name=class_name,
                bounding_box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            )


def _predict_plate_regions(image: np.ndarray, vehicle_bbox: BoundingBox, model: YOLO) -> list[BoundingBox]:
    """Detect license plate regions inside a vehicle bounding box.

    Parameters
    ----------
    image : numpy.ndarray
        Full frame (BGR) image.
    vehicle_bbox : BoundingBox
        Bounding box for the vehicle crop within which plates are searched.
    model : ultralytics.YOLO
        YOLO plate detection model.

    Returns
    -------
    list[BoundingBox]
        List of detected plate bounding boxes in absolute frame coordinates. Empty if none.
    """
    x_min_v, y_min_v, x_max_v, y_max_v = vehicle_bbox.as_xyxy()
    crop = image[y_min_v:y_max_v, x_min_v:x_max_v]
    predictions = model.predict(crop, verbose=False, max_det=1)  # only highest confidence plate
    plate_bboxes: list[BoundingBox] = []
    for result in predictions:
        for box in result.boxes:
            x_min_r, y_min_r, x_max_r, y_max_r = (
                int(box.xyxy[0][0]),
                int(box.xyxy[0][1]),
                int(box.xyxy[0][2]),
                int(box.xyxy[0][3]),
            )
            plate_bboxes.append(
                BoundingBox(
                    x_min=x_min_v + x_min_r,
                    y_min=y_min_v + y_min_r,
                    x_max=x_min_v + x_max_r,
                    y_max=y_min_v + y_max_r,
                )
            )
    return plate_bboxes


def _prepare_plate_roi(roi: np.ndarray, min_area: int) -> np.ndarray:
    """Prepare a plate region for OCR (upscale + normalize).

    Parameters
    ----------
    roi : numpy.ndarray
        Raw plate region (BGR or grayscale) extracted from the frame.
    min_area : int
        Minimum area threshold; the region is iteratively upscaled (x2) until this area is reached.

    Returns
    -------
    numpy.ndarray
        Pre-processed grayscale image optimised for OCR.
    """
    treated = roi.copy()
    while treated.shape[0] * treated.shape[1] < min_area:
        treated = cv2.resize(treated, (2 * treated.shape[1], 2 * treated.shape[0]))
    norm_img = np.zeros((treated.shape[0], treated.shape[1]))
    treated = cv2.normalize(treated, norm_img, 0, 255, cv2.NORM_MINMAX)
    treated = cv2.threshold(treated, 127, 255, cv2.THRESH_BINARY)[1]
    treated = cv2.GaussianBlur(treated, (1, 1), 0)
    return treated


def _ocr_plate(
    image: np.ndarray, pytesseract_cli_config: str, confidence_threshold: float
) -> tuple[str | None, float | None]:
    """Perform OCR on a prepared license plate image.

    Parameters
    ----------
    image : numpy.ndarray
        Pre-processed (binarised / normalised) plate image.
    pytesseract_cli_config : str
        CLI arguments forwarded to Tesseract (e.g. --psm 7).
    confidence_threshold : float
        Minimum confidence (0-1) for a token to be considered valid.

    Returns
    -------
    tuple[str | None, float | None]
        Extracted text (first token) and its confidence, or (None, None) if nothing valid detected.
    """
    data = pytesseract.image_to_data(image, config=pytesseract_cli_config, output_type="data.frame")
    data = data[data["conf"] > confidence_threshold][["conf", "text"]]  # filter very low confidence noise
    if data.empty:
        return None, None
    first = data.iloc[0]
    return str(first["text"]), float(first["conf"])


def _detect_plate(
    image: np.ndarray, yolo_object_model: YOLO, yolo_plate_model: YOLO, config: AppConfig
) -> list[DetectedObject]:
    """Run full detection pipeline for a single frame.

    Performs vehicle tracking, plate localisation, ROI preprocessing and OCR.

    Parameters
    ----------
    image : numpy.ndarray
        Input frame (BGR) from the video source.
    yolo_object_model : ultralytics.YOLO
        YOLO model used for vehicle detection / tracking.
    yolo_plate_model : ultralytics.YOLO
        YOLO model used for license plate detection.
    config : AppConfig
        Application configuration (OCR thresholds & model paths).

    Returns
    -------
    list[DetectedObject]
        List of detected objects with optional OCR results (empty if nothing detected).
    """
    logger.debug(f"Starting object tracking on frame of shape {image.shape}")
    detected: list[DetectedObject] = []
    for vehicle in _track_vehicles(image, yolo_object_model):
        logger.trace(f"Vehicle detected id={vehicle.object_id} class={vehicle.class_name} bbox={vehicle.bounding_box}")
        plate_bboxes = _predict_plate_regions(image, vehicle.bounding_box, yolo_plate_model)
        if not plate_bboxes:
            # Record vehicle even if no plate detected
            detected.append(
                DetectedObject(
                    object_id=vehicle.object_id,
                    license_plate_text=None,
                    license_plate_confidence=None,
                    predicted_object_type=vehicle.class_name,
                    object_bounding_box=vehicle.bounding_box,
                    plate_bounding_box=None,
                )
            )
            continue
        for plate_bbox in plate_bboxes:
            x_min_p, y_min_p, x_max_p, y_max_p = plate_bbox.as_xyxy()
            roi = image[y_min_p:y_max_p, x_min_p:x_max_p]
            prepared = _prepare_plate_roi(roi, config.license_plate_ocr.resizing_threshold)
            plate_text, plate_conf = _ocr_plate(
                prepared,
                config.license_plate_ocr.pytesseract_config,
                config.license_plate_ocr.confidence_threshold,
            )
            logger.debug(
                f"Plate candidate object_id={vehicle.object_id} plate_bbox={plate_bbox.as_xyxy()} text={plate_text} conf={plate_conf}",
            )
            detected.append(
                DetectedObject(
                    object_id=vehicle.object_id,
                    license_plate_text=plate_text,
                    license_plate_confidence=plate_conf,
                    predicted_object_type=vehicle.class_name,
                    object_bounding_box=vehicle.bounding_box,
                    plate_bounding_box=plate_bbox,
                )
            )
    logger.debug(f"Frame processing complete: {len(detected)} objects detected")
    return detected


def _plot_objects_in_image(
    image: np.ndarray, detected_objects: list[DetectedObject], line_thickness: int = 5, font_scale: int = 2
) -> np.ndarray:
    """Annotate a frame with detected vehicles and license plates.

    Parameters
    ----------
    image : numpy.ndarray
        Original BGR frame.
    detected_objects : list[DetectedObject]
        Detected objects to render (vehicle + plate bounding boxes, OCR text omitted for privacy).
    line_thickness : int, default=5
        Thickness of rectangle borders.
    font_scale : int, default=2
        Scale factor for rendered text labels.

    Returns
    -------
    numpy.ndarray
        Copy of the input frame with rectangles and labels drawn.

    Notes
    -----
    OCR text is intentionally not rendered; adjust here if plate text display becomes a requirement.
    """

    logger.trace(f"Annotating {len(detected_objects)} objects on frame")
    image_with_annotations = image.copy()

    for detected_object in detected_objects:
        object_bb_t = detected_object.object_bounding_box
        object_bb = (object_bb_t.x_min, object_bb_t.y_min, object_bb_t.x_max, object_bb_t.y_max)
        # Vehicle box
        cv2.rectangle(
            image_with_annotations,
            (object_bb[0], object_bb[1]),
            (object_bb[2], object_bb[3]),
            (255, 0, 0),
            line_thickness,
        )
        cv2.putText(
            image_with_annotations,
            f"ID:{detected_object.object_id} {detected_object.predicted_object_type}",
            (object_bb[0], max(0, object_bb[1] - 10)),
            cv2.FONT_HERSHEY_PLAIN,
            font_scale,
            (255, 0, 0),
            font_scale,
        )
        # Plate box (if available)
        if detected_object.plate_bounding_box is not None:
            plate_bb_t = detected_object.plate_bounding_box
            plate_bb = (plate_bb_t.x_min, plate_bb_t.y_min, plate_bb_t.x_max, plate_bb_t.y_max)
            cv2.rectangle(
                image_with_annotations,
                (plate_bb[0], plate_bb[1]),
                (plate_bb[2], plate_bb[3]),
                (255, 0, 0),
                line_thickness,
            )
            cv2.putText(
                image_with_annotations,
                "plate",
                (plate_bb[0], max(0, plate_bb[1] - 10)),
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
    """Run synchronous inference and produce output artifacts.

    Parameters
    ----------
    frame : numpy.ndarray
        Raw BGR frame.
    frame_index : int
        1-based index of the frame within the stream.
    yolo_object_model : ultralytics.YOLO
        YOLO model for vehicle detection/tracking.
    yolo_plate_model : ultralytics.YOLO
        YOLO model for plate detection.
    config : AppConfig
        Application configuration (OCR thresholds etc.).

    Returns
    -------
    tuple[list[DetectedObject], bytes, dict]
        Detected objects, encoded JPEG bytes of the annotated frame, and a JSON-serialisable metadata payload.
    """
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
    """Stream frames + metadata as multipart/mixed HTTP body.

    Each frame yields two consecutive parts sharing the same processing results:

    1. JSON metadata (detections + frame index)
    2. JPEG image (annotated frame)

    The boundary token is --frame and the stream terminates with --frame--.

    Parameters
    ----------
    video : cv2.VideoCapture
        OpenCV video capture object (already opened).
    yolo_object_model : ultralytics.YOLO
        YOLO model for vehicle detection/tracking.
    yolo_plate_model : ultralytics.YOLO
        YOLO model for plate detection.
    config : AppConfig
        Application configuration.

    Yields
    ------
    bytes
        Multipart chunk (headers + payload). Caller is responsible for writing directly to the HTTP response.
    """
    boundary = b"--frame"
    frame_index = 0
    while True:
        ok, frame = video.read()
        if not ok:
            logger.info(f"Multipart stream ended at frame_index={frame_index}")
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
        except (cv2.error, TesseractError) as exc:
            logger.exception(f"Error processing frame_index={frame_index}: {exc}")
            continue
    yield b"--frame--\r\n"
