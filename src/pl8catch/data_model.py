"""Data Model of the current implementation"""

from pydantic import BaseModel
import yaml


class LicensePlateOCRConfig(BaseModel):
    resizing_threshold: int
    pytesseract_config: str


class ModelsConfig(BaseModel):
    object_detection: str
    license_plate: str


class YAMLConfig(BaseModel):
    license_plate_ocr: LicensePlateOCRConfig
    models: ModelsConfig


def read_yaml(file_path: str) -> YAMLConfig:
    with open(file_path, "r") as stream:
        config = yaml.safe_load(stream)
    return YAMLConfig(**config)


class DetectedObject(BaseModel):
    object_id: int | None  # Sometimes YOLO is returning None when its not relevant
    license_plate_text: str | None  # License plate might not be detected or recognized
    license_plate_confidence: float | None  # License plate might not be detected or recognized
    predicted_object_type: str
    object_biding_box: tuple[int, int, int, int]
    plate_biding_box: tuple[int, int, int, int]


CONFIG = read_yaml("config.yaml")
