"""FastAPI backend of pl8catch"""

from importlib.metadata import version
from loguru import logger
from fastapi.responses import StreamingResponse
import uvicorn
import cv2
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException


from pl8catch.lifespan import lifespan
from pl8catch.data_model import AppConfig, VideoSource
from pl8catch.utils import stream_frame_and_detections_multipart
from pl8catch.environment import Environment
from pl8catch.log import configure_logging

env: Environment = Environment()
config: AppConfig = AppConfig.from_file(env.CONFIG_FILE_PATH)

configure_logging(env.LOG_LEVEL)

yolo_object_model = YOLO(config.models.object_detection)
yolo_plate_model = YOLO(config.models.license_plate)

app = FastAPI(
    title="Pl8Catch",
    version=version("pl8catch"),
    summary="A API for vehicle detection and license plate recognition.",
    lifespan=lifespan,
    contact={
        "name": "Daniel Sousa",
        "email": "danielsoussa@gmail.com",
    },
)


@app.post("/video-detection", summary="Stream annotated frames + detections (multipart)")
async def video_detection_endpoint(payload: VideoSource) -> StreamingResponse:
    """Stream annotated frames and detection metadata via multipart/mixed.

    For each processed frame two parts are emitted with boundary 'frame':
      1. JSON metadata (application/json)
      2. Annotated JPEG (image/jpeg)

    Parameters
    ----------
    payload : VideoSource
        The request object containing the video source URL or file path.

    Returns
    -------
    StreamingResponse
        A streaming response with media type multipart/mixed.

    Raises
    ------
    HTTPException
        If the video source cannot be opened.

    """
    source = payload.source
    logger.info("Starting multipart video detection stream for source: %s", source)
    video = cv2.VideoCapture(source)
    if not video.isOpened():
        raise HTTPException(status_code=400, detail=f"Cannot open video source: {source}")

    return StreamingResponse(
        stream_frame_and_detections_multipart(video, yolo_object_model, yolo_plate_model, config),
        media_type="multipart/mixed; boundary=frame",
    )


if __name__ == "__main__":
    uvicorn.run(app, port=config.server.port, host=config.server.host)
