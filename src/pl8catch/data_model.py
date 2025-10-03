"""Data model and configuration loader for pl8catch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar, cast

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


TConfig = TypeVar("TConfig", bound="BaseFileConfig")


class BaseFileConfig(BaseModel):
    """Utility mixin that adds file-based initialisation for Pydantic models."""

    @classmethod
    def from_file(cls: type[TConfig], file_path: str | Path) -> TConfig:
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as stream:
            raw_config: Any = yaml.safe_load(stream) or {}
        return cast(TConfig, cls.model_validate(raw_config))


class LicensePlateOCRConfig(BaseModel):
    resizing_threshold: int = Field(description="Minimum area before upscaling the plate region.")
    pytesseract_config: str = Field(description="CLI flags forwarded to pytesseract for OCR.")

    model_config = ConfigDict(extra="forbid")


class ModelsConfig(BaseModel):
    object_detection: str = Field(description="Path to the YOLO model used for vehicle detection.")
    license_plate: str = Field(description="Path to the YOLO model used for license plate detection.")

    model_config = ConfigDict(extra="forbid")


class ServerConfig(BaseModel):
    host: str = Field("127.0.0.1", description="Host to bind to.")
    port: int = Field(8000, description="Port to bind to.")

    model_config = ConfigDict(extra="forbid")


class AppConfig(BaseFileConfig):
    license_plate_ocr: LicensePlateOCRConfig = Field(
        description="Configuration parameters for license plate preprocessing and OCR.",
        validation_alias=AliasChoices("license_plate_ocr", "licensePlateOCR"),
    )
    models: ModelsConfig = Field(description="File paths for YOLO models used across the application.")
    server: ServerConfig = Field(description="Server host and port configuration.")

    model_config = ConfigDict(extra="allow")


class DetectedObject(BaseModel):
    object_id: int | None  # Sometimes YOLO is returning None when its not relevant
    license_plate_text: str | None  # License plate might not be detected or recognized
    license_plate_confidence: float | None  # License plate might not be detected or recognized
    predicted_object_type: str
    object_biding_box: tuple[int, int, int, int]
    plate_biding_box: tuple[int, int, int, int]
