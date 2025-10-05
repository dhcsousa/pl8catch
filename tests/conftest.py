from pathlib import Path
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest
import requests
from pydantic import HttpUrl

from pl8catch.config import AppConfig
from pl8catch.config.env import Environment


def _resolve_model_path(value: HttpUrl | Path) -> Path:
    if isinstance(value, HttpUrl):
        env = Environment()
        filename = Path(value.path).name
        return env.MODEL_DIR / filename
    return Path(value)


@pytest.fixture(scope="session")
def config() -> AppConfig:
    return AppConfig.from_file("configs/backend.yaml")


@pytest.fixture(scope="session", autouse=True)
def _patch_yolo_plate(config: AppConfig):
    plate_path = _resolve_model_path(config.models.license_plate)
    if plate_path.exists():
        return  # real weights available – do nothing keep real YOLO model

    # Else use dummy model as plate detector
    import ultralytics  # type: ignore
    from ultralytics import YOLO as RealYOLO  # type: ignore

    import numpy as _np

    class _DummyPlateBox:
        def __init__(self):
            self.xyxy = _np.array([[0, 0, 60, 36]])
            self.id = None
            self.cls = _np.array([0])

    class _DummyPlateResult:
        def __init__(self):
            self.boxes = [_DummyPlateBox()]
            self.names = {0: "plate"}

    class _DummyPlateModel:
        def predict(self, *_, **__):
            return [_DummyPlateResult()]

    # Subclass wrapper to intercept creation for the missing plate path
    class PatchedYOLO(RealYOLO):
        def __new__(cls, model="yolo_stub.pt", *a, **k):
            if isinstance(model, (str, Path)) and str(model) == str(plate_path):
                return _DummyPlateModel()
            return RealYOLO(model, *a, **k)

    ultralytics.YOLO = PatchedYOLO


@pytest.fixture(scope="session")
def models():
    import numpy as _np

    class _VehicleBox:
        def __init__(self):
            self.id = _np.array([1])
            self.cls = _np.array([2])
            self.xyxy = _np.array([[0, 0, 100, 60]])

    class _VehicleResult:
        def __init__(self):
            self.boxes = [_VehicleBox()]
            self.names = {2: "car"}

    class DummyVehicleModel:
        def track(self, *_a, **_k):
            return [_VehicleResult()]

    class _PlateBox:
        def __init__(self):
            self.xyxy = _np.array([[10, 10, 60, 40]])

    class _PlateResult:
        def __init__(self):
            self.boxes = [_PlateBox()]

    class DummyPlateModel:
        def predict(self, *_a, **_k):
            return [_PlateResult()]

    return DummyVehicleModel(), DummyPlateModel()


@pytest.fixture()
@pytest.mark.network
def test_image() -> np.ndarray:
    url = "https://hips.hearstapps.com/hmg-prod/images/rs357223-1626456047.jpg"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    img_np = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(img_np, cv2.IMREAD_COLOR)


@pytest.fixture()
def single_frame_video(tmp_path, test_image: np.ndarray) -> Path:
    """Create a temporary single-frame MJPG video for streaming tests."""
    video_path = tmp_path / "single_frame.avi"
    h, w = test_image.shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 1.0, (w, h))
    for _ in range(2):  # a couple frames
        writer.write(test_image)
    writer.release()
    return video_path


@pytest.fixture()
def client():
    from pl8catch.app import app

    with TestClient(app) as c:
        yield c
