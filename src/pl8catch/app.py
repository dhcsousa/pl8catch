"""FastAPI backend of pl8catch"""

import base64
import json
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse
import uvicorn
import cv2
from ultralytics import YOLO
from fastapi import FastAPI


from pl8catch.data_model import CONFIG
from pl8catch.utils import detect_plate, plot_objects_in_image


yolo_object_model = YOLO(CONFIG.models.object_detection)
yolo_plate_model = YOLO(CONFIG.models.license_plate)

app = FastAPI(
    title="Pl8Catch",
    version="0.0.1",
    summary="A API for vehicle detection and license plate recognition.",
)

video_address = "demo_files/demo_video_trimmed.mp4"
video = cv2.VideoCapture(video_address)


async def get_video_frame(video: cv2.VideoCapture) -> AsyncGenerator:
    """
    Stream video frames with object and license plate detection.

    Parameters
    ----------
    video : cv2.VideoCapture
        A cv2.VideoCapture object to capture video frames.

    Returns
    ----------
    Generator[bytes, None, None]
        A generator yielding video frames as JPEG-encoded bytes.

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
        detected_objects = detect_plate(frame, yolo_object_model, yolo_plate_model, CONFIG)
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


@app.get("/video-detection")
async def video_detection_endpoint() -> StreamingResponse:
    """
    FastAPI endpoint for video detection.

    Returns
    ----------
    StreamingResponse
        A streaming response with the video frames, encoded as JPEG images and streamed in a format
        compatible with MJPEG (multipart/x-mixed-replace).

    Notes
    ----------
    This endpoint captures video frames from a predefined video file, detects objects and license plates
    in each frame, annotates the frames, and streams them to the client. The stream is intended for real-time
    viewing in a web browser.
    """
    return StreamingResponse(get_video_frame(video), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="localhost")
