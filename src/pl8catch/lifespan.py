from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from loguru import logger
from ultralytics import YOLO

from pl8catch.config import AppConfig, Environment


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    """Application lifespan context.

    Responsibilities
    -----------------
    * Build and cache Environment & AppConfig.
    * Load YOLO models once and attach to app.state.
    * Log startup / shutdown events.
    """

    env = Environment()
    app.state.env = env

    # Application configuration
    logger.info(f"Using configuration file at {str(env.CONFIG_FILE_PATH)}.")
    config: AppConfig = AppConfig.from_file(env.CONFIG_FILE_PATH)
    app.state.config = config

    # Lazy / centralized model loading
    logger.info(f"Loading YOLO object detection model from: {config.models.object_detection}")
    yolo_object_model = YOLO(config.models.object_detection)
    logger.info(f"Loading YOLO license plate model from: {config.models.license_plate}")
    yolo_plate_model = YOLO(config.models.license_plate)
    app.state.yolo_object_model = yolo_object_model
    app.state.yolo_plate_model = yolo_plate_model
    logger.info("YOLO models loaded successfully.")

    logger.info("Starting pl8catch backend server...")

    yield

    logger.info("Stopping pl8catch backend server...")
