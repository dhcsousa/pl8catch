"""Configuration models for pl8catch."""

from pathlib import Path
from typing import Any, TypeVar, cast

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

TConfig = TypeVar("TConfig", bound="BaseFileConfig")


class BaseFileConfig(BaseModel):
    """Base class for file-based configuration models.

    Methods
    -------
    from_file(file_path: str | Path) -> TConfig
        Load configuration from a YAML file and return an instance of the model.
    """

    @classmethod
    def from_file(cls: type[TConfig], file_path: str | Path) -> TConfig:  # pragma: no cover - thin IO wrapper
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as stream:
            raw_config: Any = yaml.safe_load(stream) or {}
        return cast(TConfig, cls.model_validate(raw_config))


class LicensePlateOCRConfig(BaseModel):
    """Configuration controlling preprocessing & OCR for license plates.

    Parameters
    ----------
    resizing_threshold : int
        Minimum area (width * height) required before upscaling the plate region.
    pytesseract_config : str
        CLI flags forwarded verbatim to pytesseract.
    confidence_threshold : float
        Minimum OCR token confidence (default is 0.1) for pytesseract. Helps filter noise.
    """

    resizing_threshold: int = Field(description="Minimum area before upscaling the plate region.")
    pytesseract_config: str = Field(description="CLI flags forwarded to pytesseract for OCR.")
    confidence_threshold: float = Field(
        0.1, description="Minimum OCR token confidence (0..1) to keep when parsing pytesseract results."
    )

    model_config = ConfigDict(extra="forbid")


class ModelsConfig(BaseModel):
    """Holds file paths to YOLO model weights.

    Parameters
    ----------
    object_detection : str
        Path to the YOLO model used for vehicle detection / tracking.
    license_plate : str
        Path to the YOLO model used for license plate detection.
    """

    object_detection: str = Field(description="Path to the YOLO model used for vehicle detection.")
    license_plate: str = Field(description="Path to the YOLO model used for license plate detection.")

    model_config = ConfigDict(extra="forbid")


class ServerConfig(BaseModel):
    """Server host/port configuration.

    Parameters
    ----------
    host : str
        Host/IP address to bind the application.
    port : int
        TCP port to bind.
    """

    host: str = Field("127.0.0.1", description="Host to bind to.")
    port: int = Field(8000, description="Port to bind to.")

    model_config = ConfigDict(extra="forbid")


class AppConfig(BaseFileConfig):
    """Top-level application configuration loaded from YAML.

    Parameters
    ----------
    license_plate_ocr : LicensePlateOCRConfig
        Parameters for plate region resizing and OCR CLI flags.
    models : ModelsConfig
        Paths to YOLO weight files used by the application.
    server : ServerConfig
        Server host and port configuration.
    """

    license_plate_ocr: LicensePlateOCRConfig = Field(
        description="Configuration parameters for license plate preprocessing and OCR.",
        validation_alias=AliasChoices("license_plate_ocr", "licensePlateOCR"),
    )
    models: ModelsConfig = Field(description="File paths for YOLO models used across the application.")
    server: ServerConfig = Field(description="Server host and port configuration.")

    # Allow unknown top-level keys so we can extend without breaking older files.
    model_config = ConfigDict(extra="allow")
