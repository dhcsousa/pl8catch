"""Unit tests for fetch_model helper."""

from pathlib import Path
import io

import pytest
from pydantic import HttpUrl
from requests.exceptions import HTTPError

from pl8catch.core.ml_models import fetch_model


class DummyResponse(io.BytesIO):
    """A lightweight stand-in for ``requests.Response``."""

    def __init__(self, payload: bytes, status_code: int = 200):
        super().__init__(payload)
        self.status_code = status_code

    def iter_content(self, chunk_size: int = 8192):  # pragma: no cover - trivial
        while True:
            chunk = self.read(chunk_size)
            if not chunk:
                break
            yield chunk

    def raise_for_status(self):  # pragma: no cover - trivial
        if self.status_code >= 400:
            raise HTTPError(f"HTTP error {self.status_code}")


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    return tmp_path / "models"


def test_fetch_model_downloads_when_missing(monkeypatch, model_dir):
    url: HttpUrl = HttpUrl("https://example.com/models/yolo.pt")
    payload = b"binary-weights-data"

    def fake_get(u, timeout, stream):  # noqa: ARG001
        assert u == str(url)
        assert timeout == 30
        assert stream is True
        return DummyResponse(payload)

    monkeypatch.setattr("pl8catch.core.ml_models.get", fake_get)
    result = fetch_model(url, model_dir)
    assert result.exists()
    assert result.read_bytes() == payload


def test_fetch_model_uses_cache(monkeypatch, model_dir):
    url: HttpUrl = HttpUrl("https://example.com/models/vehicle.pt")
    # Pre-create the file to simulate cache hit
    cached = model_dir / "vehicle.pt"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"cached")

    called = False

    def fake_get(u, timeout, stream):  # noqa: ANN001, ARG001
        nonlocal called
        called = True
        return DummyResponse(b"new")

    monkeypatch.setattr("pl8catch.core.ml_models.get", fake_get)
    result = fetch_model(url, model_dir)
    assert result == cached
    assert called is False, "Should not download when file already exists"


def test_fetch_model_local_path_success(tmp_path: Path, model_dir: Path):
    local_model = tmp_path / "local.pt"
    local_model.write_text("weights")
    result = fetch_model(local_model, model_dir)
    assert result == local_model


def test_fetch_model_local_path_missing(tmp_path: Path, model_dir: Path):
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError):
        fetch_model(missing, model_dir)
