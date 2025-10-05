"""Inference / streaming routes."""

import cv2
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from ultralytics import YOLO

from pl8catch.config import AppConfig
from pl8catch.core.domain_models import VideoSource
from pl8catch.core.utils import stream_frame_and_detections_multipart

router = APIRouter()


@router.post("/video-detection", summary="Stream annotated frames + detections (multipart)")
async def video_detection_endpoint(request: Request, payload: VideoSource) -> StreamingResponse:
    """Stream annotated frames and detection metadata via multipart/mixed.

    Parameters
    ----------
    payload : VideoSource
        Request body containing the video source path/URI.
    """
    config: AppConfig = request.app.state.config
    try:
        yolo_object_model: YOLO = request.app.state.yolo_object_model
        yolo_plate_model: YOLO = request.app.state.yolo_plate_model
    except AttributeError:  # pragma: no cover - defensive
        logger.error("YOLO models not present on app.state. Startup may have failed.")
        raise HTTPException(status_code=500, detail="Models not loaded.")
    source = payload.source
    logger.info(f"Starting multipart video detection stream for source: {source}")
    video = cv2.VideoCapture(source)
    if not video.isOpened():
        raise HTTPException(status_code=400, detail=f"Cannot open video source: {source}")

    return StreamingResponse(
        stream_frame_and_detections_multipart(video, yolo_object_model, yolo_plate_model, config),
        media_type="multipart/mixed; boundary=frame",
    )
