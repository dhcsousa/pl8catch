import os
from pathlib import Path
import pytest
from pydantic import HttpUrl

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
      object_detection: "yolo_a.pt"
      license_plate: "yolo_b.pt"
    """
    return yaml_data


def test_read_yaml(sample_yaml_data, tmp_path):
    # Write sample YAML data to a temp file using pytest's tmp_path fixture
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(sample_yaml_data)

    config = AppConfig.from_file(cfg_file)

    # Check if the returned object is of type AppConfig
    assert isinstance(config, AppConfig)

    # Check nested objects' types and values
    assert isinstance(config.license_plate_ocr, LicensePlateOCRConfig)
    assert config.license_plate_ocr.resizing_threshold == 100
    assert config.license_plate_ocr.pytesseract_config == "random"

    assert isinstance(config.models, ModelsConfig)
    assert config.models.object_detection == Path("yolo_a.pt")
    assert config.models.license_plate == Path("yolo_b.pt")


def test_environment_defaults(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")  # Override for this test
    env = Environment()  # load defaults
    assert env.LOG_LEVEL == "DEBUG"
    assert env.CONFIG_FILE_PATH.name == "backend.yaml"
    # Ensure root dir resolves up to project root (contains pyproject.toml)
    assert (env.ROOT_DIR / "pyproject.toml").exists()


def test_server_section_optional(tmp_path):
    """Config files without a 'server' section should still validate using defaults."""
    yaml_data = """
    license_plate_ocr:
      resizing_threshold: 123
      pytesseract_config: "foo"
    models:
      object_detection: "obj.pt"
      license_plate: "lp.pt"
    """
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml_data)
    config = AppConfig.from_file(cfg_file)
    assert config.server.host == "127.0.0.1"
    assert config.server.port == 8000


def test_models_remote_url_uses_httpurl(tmp_path):
    yaml_data = """
    license_plate_ocr:
      resizing_threshold: 456
      pytesseract_config: "bar"
    models:
      object_detection: "https://example.com/model.pt"
      license_plate: "lp.pt"
    """
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml_data)
    config = AppConfig.from_file(cfg_file)
    assert isinstance(config.models.object_detection, HttpUrl)
    assert str(config.models.object_detection) == "https://example.com/model.pt"
    assert isinstance(config.models.license_plate, Path)
