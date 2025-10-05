"""Model resolution & download helpers"""

from pathlib import Path

from loguru import logger
from pydantic import HttpUrl
from requests import get


def fetch_model(model_path: Path | HttpUrl, model_dir: Path) -> Path:
    """Ensure a model is present locally, downloading if necessary.

    Parameters
    ----------
    model_path : Path | HttpUrl
        Local path or URL to the model weights.

    model_dir : Path
        Directory to store the model weights.

    Returns
    -------
    Path
        Local path to the model weights.

    """
    if isinstance(model_path, HttpUrl):
        url = str(model_path)
        filename = Path(model_path.path).name
        dest = model_dir / filename
        if dest.exists():
            logger.info(f"Model file {filename} already present at {dest}. Using this copy.")
            return dest
        logger.info(f"Downloading model from {url} -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)

        r = get(url, timeout=30, stream=True)
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return dest

    else:
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        logger.info(f"Using local model path: {model_path}")
        return model_path
