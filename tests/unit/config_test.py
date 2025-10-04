import os
import tempfile

import pytest
from pl8catch.config.app_config import AppConfig, LicensePlateOCRConfig, ModelsConfig
from pl8catch.config.env import Environment


@pytest.fixture()
def sample_yaml_data():
    # Sample YAML data for testing
    yaml_data = """
    server:
      host: "127.0.0.1"
      port: 8000
    license_plate_ocr:
      resizing_threshold: 100
      pytesseract_config: "random"
    models:
      object_detection: "yolo_a"
      license_plate: "yolo_b"
    """
    return yaml_data


def test_read_yaml(sample_yaml_data):
    # Create a temporary file with sample YAML data
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
        tmp_file.write(sample_yaml_data)
        tmp_file_path = tmp_file.name

    try:
        # Call the function with the temporary file path
        config = AppConfig.from_file(tmp_file_path)

        # Check if the returned object is of type AppConfig
        assert isinstance(config, AppConfig)

        # Check if the nested objects are of correct types and have the correct values
        assert isinstance(config.license_plate_ocr, LicensePlateOCRConfig)
        assert config.license_plate_ocr.resizing_threshold == 100
        assert config.license_plate_ocr.pytesseract_config == "random"

        assert isinstance(config.models, ModelsConfig)
        assert config.models.object_detection == "yolo_a"
        assert config.models.license_plate == "yolo_b"

    finally:
        # Delete the temporary file
        os.unlink(tmp_file_path)


def test_environment_defaults():
    env = Environment()  # load defaults
    assert env.LOG_LEVEL == "DEBUG"
    assert env.CONFIG_FILE_PATH.name == "backend.yaml"
    # Ensure root dir resolves up to project root (contains pyproject.toml)
    assert (env.ROOT_DIR / "pyproject.toml").exists()
