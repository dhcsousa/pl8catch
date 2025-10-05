"""Unit tests for fetch_model helper."""

from pathlib import Path
import io
import urllib.request

import pytest
from pydantic import HttpUrl

from pl8catch.core.ml_models import fetch_model


class DummyResponse(io.BytesIO):
    """A simple context manager mimicking urllib response."""

    def __enter__(self):  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - trivial
        self.close()
        return False


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    return tmp_path / "models"


def test_fetch_model_downloads_when_missing(monkeypatch, model_dir):
    url: HttpUrl = HttpUrl("https://example.com/models/yolo.pt")
    payload = b"binary-weights-data"

    def fake_urlopen(u):
        assert str(u) == str(url)
        return DummyResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
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

    def fake_urlopen(u):  # noqa: ANN001
        nonlocal called
        called = True
        return DummyResponse(b"new")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
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
