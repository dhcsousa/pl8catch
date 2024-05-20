import os
import tempfile
import pytest

from pl8catch.data_model import LicensePlateOCRConfig, ModelsConfig, YAMLConfig, read_yaml


@pytest.fixture()
def sample_yaml_data():
    # Sample YAML data for testing
    yaml_data = """
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
        config = read_yaml(tmp_file_path)

        # Check if the returned object is of type YAMLConfig
        assert isinstance(config, YAMLConfig)

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
