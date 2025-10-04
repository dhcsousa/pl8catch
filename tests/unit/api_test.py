from fastapi.testclient import TestClient
from pathlib import Path


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_config(client: TestClient):
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "models" in data
    assert "license_plate_ocr" in data


def test_inference_no_image(client: TestClient):
    # Provide an invalid video source path; endpoint should return 400
    payload = {"source": "non_existent_video_file.mp4"}
    response = client.post("/video-detection", json=payload)
    assert response.status_code == 400, response.text
    data = response.json()
    assert "Cannot open video source" in data["detail"]


def test_inference_with_image(client: TestClient, single_frame_video: Path):
    payload = {"source": str(single_frame_video)}
    response = client.post("/video-detection", json=payload)
    assert response.status_code == 200, response.text
    content_type = response.headers.get("content-type", "")
    assert content_type.startswith("multipart/mixed")
    iterator = response.iter_bytes()
    first_chunk = b""
    try:
        for _ in range(5):
            first_chunk += next(iterator)
            if b"--frame" in first_chunk and len(first_chunk) > 50:
                break
    except StopIteration:  # pragma: no cover
        pass
    assert first_chunk, "Expected non-empty stream chunk"
    assert b"--frame" in first_chunk
