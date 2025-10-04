from fastapi.testclient import TestClient
import cv2
import numpy as np
import tempfile
from pathlib import Path


def test_heatlh(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_config(client: TestClient):
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "server" in data
    assert "models" in data
    assert "license_plate_ocr" in data


def test_inference_no_image(client: TestClient):
    # Provide an invalid video source path; endpoint should return 400
    payload = {"source": "non_existent_video_file.mp4"}
    response = client.post("/video-detection", json=payload)
    assert response.status_code == 400, response.text
    data = response.json()
    assert "Cannot open video source" in data["detail"]


def test_inference_with_image(client: TestClient, test_image: np.ndarray):
    # Write the provided test image to a temporary video file (single-frame video)
    height, width = test_image.shape[:2]
    with tempfile.TemporaryDirectory() as td:
        video_path = Path(td) / "single_frame.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(video_path), fourcc, 1.0, (width, height))
        # Write a couple frames to satisfy container expectations
        for _ in range(2):
            writer.write(test_image)
        writer.release()

        payload = {"source": str(video_path)}
        response = client.post("/video-detection", json=payload)
        # Expect a 200 with multipart/mixed stream
        assert response.status_code == 200, response.text
        content_type = response.headers.get("content-type", "")
        assert content_type.startswith("multipart/mixed")
        # Consume a few bytes from the streaming iterator to confirm data starts flowing.
        iterator = response.iter_bytes()  # FastAPI TestClient provides iter_bytes for streaming responses
        first_chunk = b""
        try:
            for _ in range(5):  # try a few chunks to get some data
                first_chunk += next(iterator)
                if b"--frame" in first_chunk and len(first_chunk) > 50:
                    break
        except StopIteration:  # pragma: no cover - defensive
            pass
        assert first_chunk, "Expected non-empty stream chunk"
        assert b"--frame" in first_chunk, "Multipart boundary not found in initial stream bytes"
