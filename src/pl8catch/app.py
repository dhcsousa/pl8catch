"""FastAPI backend of pl8catch"""

from fastapi.responses import StreamingResponse
import uvicorn
import cv2
from ultralytics import YOLO
from fastapi import FastAPI


from pl8catch.data_model import CONFIG
from pl8catch.utils import stream_detected_frames


yolo_object_model = YOLO(CONFIG.models.object_detection)
yolo_plate_model = YOLO(CONFIG.models.license_plate)

app = FastAPI(
    title="Pl8Catch",
    version="0.0.1",
    summary="A API for vehicle detection and license plate recognition.",
)

video_address = "demo_files/demo_video_trimmed.mp4"
video = cv2.VideoCapture(video_address)


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
    return StreamingResponse(
        stream_detected_frames(video, yolo_object_model, yolo_plate_model, CONFIG), media_type="text/event-stream"
    )


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="localhost")
