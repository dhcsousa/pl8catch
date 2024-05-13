from typing import Any
import cv2
import pytesseract
import numpy as np
from ultralytics import YOLO


class DetectedObject:
    def __init__(
        self,
        license_plate_text: str,
        license_plate_confidence: float,
        predicted_object_type: str,
        object_biding_box: tuple[int, int, int, int],
        detected_plates: list[tuple[int, int, int, int]],
    ):
        self.license_plate_text: str = license_plate_text
        self.license_plate_confidence: float = license_plate_confidence
        self.predicted_object_type: str = predicted_object_type
        self.object_biding_box: tuple[int, int, int, int] = object_biding_box
        self.detected_plates: list[tuple[int, int, int, int]] = detected_plates


def detect_plate(
    image: np.ndarray, yolo_model: YOLO, yolo_plate_model: YOLO, config: dict[str, Any]
) -> list[DetectedObject]:
    """
    This function takes an image, two YOLO models (one for general object detection, another for license plate detection),
    and a configuration dictionary as input and returns a list of `DetectedObject` objects.

    Parameters
    ----------
    image : numpy.ndarray
        A NumPy array representing the image to be processed.
    yolo_model : YOLO model instance
        A YOLO model instance used for general object detection (e.g., cars, motorcycles).
    yolo_plate_model : YOLO model instance
        A YOLO model instance trained specifically for license plate detection.
    config : dict
        A dictionary containing configuration parameters for various steps (e.g., license plate OCR resizing threshold).

    Returns
    ----------
    list of DetectedObject objects
        A list of `DetectedObject` objects. Each object contains:
        - `license_plate_text`: A list of recognized characters from the license plate (might be empty if not detected).
        - `license_plate_confidence`: A list of confidence scores for each recognized character (might be empty if not detected).
        - `predicted_object_type`: The type of object predicted by the general YOLO model (e.g., "car").
        - `object_biding_box`: A tuple representing the bounding box of the detected object (x_min, y_min, x_max, y_max).
        - `detected_plates`: A list of tuples representing the bounding boxes of detected license plates within the object (x_min, y_min, x_max, y_max).
    """

    yolo_predictions = yolo_model.predict(image, classes=[2, 3, 5, 7], verbose=False)  # car, motorcycle, bus, truck
    detected_objects = []

    for result in yolo_predictions:
        if result.boxes:
            for box in result.boxes:
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
                image[y_min_vehicle:y_max_vehicle, x_min_vehicle:x_max_vehicle], verbose=False
            )  # Predict plates only within the car box
            detected_plates = []
            for plate in plates:
                for box in plate.boxes:
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

                    detected_plates.append((x_min_plate, y_min_plate, x_max_plate, y_max_plate))
                    # TODO: Possible improvement, only append to detected_plate, after text is detected?

                    # OCR the license plate
                    original_roi = image[y_min_plate:y_max_plate, x_min_plate:x_max_plate]

                    # I noticed that if the resolution/area of the license plate is small results degrade, increasing it improves them
                    treated_roi = original_roi.copy()
                    while (
                        treated_roi.shape[0] * treated_roi.shape[1] < config["license_plate_ocr"]["resizing_threshold"]
                    ):
                        treated_roi = cv2.resize(treated_roi, (2 * treated_roi.shape[1], 2 * treated_roi.shape[0]))

                    # Normalizing and thresholding image
                    norm_img = np.zeros((treated_roi.shape[0], treated_roi.shape[1]))
                    treated_roi = cv2.normalize(treated_roi, norm_img, 0, 255, cv2.NORM_MINMAX)
                    treated_roi = cv2.threshold(treated_roi, 127, 255, cv2.THRESH_BINARY)[1]
                    treated_roi = cv2.GaussianBlur(treated_roi, (1, 1), 0)

                    # Extract Data from License Plate Image Segment using pytesseract
                    pytesseract_data = pytesseract.image_to_data(
                        treated_roi, config=config["license_plate_ocr"]["pytesseract_config"], output_type="data.frame"
                    )
                    pytesseract_data = pytesseract_data[pytesseract_data["conf"] > 0.1][["conf", "text"]]

                    print(pytesseract_data.shape)

                    # Append the recognized character to the list
                    detected_objects.append(
                        DetectedObject(
                            pytesseract_data["text"].tolist(),
                            pytesseract_data["conf"].tolist(),
                            class_name,
                            object_biding_box,
                            detected_plates,
                        ),
                    )

                    # image_side_by_side = np.hstack((image, image_with_yolo))

                    # # Find the difference in sizes
                    # size_diff = max(treated_roi.shape[0] - original_roi.shape[0], 0)

                    # # Pad original_roi with zeros at the end to match the size of treated_roi
                    # padded_original_roi = np.pad(original_roi, ((0, size_diff), (0, 0), (0, 0)), mode="constant")

                    # # Now, you can horizontally stack them
                    # roi_side_by_side = np.hstack((padded_original_roi, treated_roi))

    return detected_objects  # image_with_yolo, image_side_by_side, roi_side_by_side


def plot_objects_in_image(image: np.ndarray, detected_objects: list[DetectedObject]) -> np.ndarray:
    """
    Plot detected objects in the input image with annotations.

    Parameters
    ----------
    image : numpy.ndarray
        A NumPy array representing the image to be plotted.
    detected_objects : list of DetectedObject objects
        A list of DetectedObject objects containing information about detected objects and their bounding boxes.

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
            2,
        )
        cv2.putText(
            image_with_annotations,
            detected_object.predicted_object_type,
            (object_bb[0], object_bb[1] - 10),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 0, 0),
            1,
        )
        for detected_plate in detected_object.detected_plates:
            cv2.rectangle(
                image_with_annotations,
                (detected_plate[0], detected_plate[1]),
                (detected_plate[2], detected_plate[3]),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                image_with_annotations,
                "plate",
                (detected_plate[0], detected_plate[1] - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                1,
            )

    return image_with_annotations
