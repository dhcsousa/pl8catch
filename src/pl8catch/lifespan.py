from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from loguru import logger
from ultralytics import YOLO

from pl8catch.config import AppConfig, Environment
from pl8catch.core.ml_models import fetch_model


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

    # Resolve / create model directory & resolve references
    object_model_path = fetch_model(config.models.object_detection, env.MODEL_DIR)
    plate_model_path = fetch_model(config.models.license_plate, env.MODEL_DIR)

    # Lazy / centralized model loading (paths now ensured to exist)
    logger.info(f"Loading YOLO object detection model from: {object_model_path}")
    yolo_object_model = YOLO(str(object_model_path))
    logger.info(f"Loading YOLO license plate model from: {plate_model_path}")
    yolo_plate_model = YOLO(str(plate_model_path))
    app.state.yolo_object_model = yolo_object_model
    app.state.yolo_plate_model = yolo_plate_model
    logger.info("YOLO models loaded successfully.")

    logger.info("Starting pl8catch backend server...")

    yield

    logger.info("Stopping pl8catch backend server...")
