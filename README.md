[![Pre-commit and Unit Tests](https://github.com/dhcsousa/pl8catch/actions/workflows/checks.yaml/badge.svg)](https://github.com/dhcsousa/pl8catch/actions/workflows/checks.yaml)

# Pl8Catch: The License Plate Tracker

Pl8Catch is a comprehensive license plate recognition system designed to detect vehicles, extract license plate information, and provide a user-friendly interface for tracking and managing license plates.

This repository serves as a compact, reproducible example of how to build a complete license plate recognition system. It includes training a YOLO model for license plate recognition, a FastAPI backend to serve the trained model and run inference, and a lightweight Streamlit frontend for demo purposes. The training process is tracked with MLflow (code, params and metrics), so you can run the provided training script and achieve similar results. If the dataset isn’t present locally, it’s downloaded automatically from Roboflow when you supply the required environment variables (at minimum `ROBOFLOW_API_KEY`).

## Features

- **YOLO Integration**: Utilizes YOLOv12 for efficient and accurate vehicle and license plate detection.
- **OCR (Optical Character Recognition)**: Employs OCR techniques to extract text from license plates.
- **FastAPI Backend**: Wraps the functionality into API using FastAPI for seamless integration into other applications.
- **Streamlit Frontend**: Provides a frontend interface built with Streamlit for easy access and interaction. The frontend is just to demonstrate the API capabilities and is not intended for production use.
- **MLflow Tracking**: Integrates MLflow for tracking experiments, metrics, and models during training.

## Training with MLflow

Ultralytics has native MLflow support, so you can keep track of metrics, parameters, and artifacts during training.

1. Start a local MLflow server (or point `MLFLOW_TRACKING_URI` to an existing server):

	```bash
	mlflow server --backend-store-uri mlflow
	```

2. Kick off training with sensible defaults and MLflow logging enabled:

	```bash
	python src/training/train.py
	```

Check the configuration file at `configs/train.yaml` (or see the `TrainConfig` class) to customize the training process by modifying parameters such as epochs, batch size, image size, and more.

**Heads up:** The script looks for the dataset at `downloaded_dataset/data.yaml`. If it cannot find the file it will attempt to download the dataset from Roboflow. Provide your own API key via `ROBOFLOW_API_KEY` environment variable. For more information visit [Roboflow](https://roboflow.com/).

You can also override `ROBOFLOW_WORKSPACE`, `ROBOFLOW_PROJECT`, `ROBOFLOW_VERSION`, and `ROBOFLOW_EXPORT_FORMAT` to point at a different dataset. Look at the config file or the `TrainConfig` class for the defaults.

## Dataset

The dataset used for training the model can be found [here](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4).

## Execution

### Task Automation (Justfile)

This repo includes a `justfile` (a lightweight alternative to a Makefile) to streamline common tasks.

Install `just` (macOS / Homebrew):
```bash
brew install just
```

List available recipes:

```bash
just
```

Typical commands (see the file for the authoritative list):

```bash
just venv        # Sync & create the virtual environment with all groups
just pre-commit  # Run formatting, linting, type checks
just test        # Run the full test suite
just clean       # Remove caches and the virtual environment (interactive confirm)
```

Using a `justfile` keeps command logic in one place and avoids remembering long invocations.

### Backend

Run the FastAPI backend:

```bash
python src/pl8catch/app.py
```

It exposes a single streaming endpoint that combines detections and annotated frames without double inference:

```
POST /video-detection
Content-Type: application/json
Body: {"source": "<video source>"}
Response: multipart/mixed; boundary=frame
```

For each processed frame the response emits TWO multipart parts under the same boundary in this order:
1. `application/json` metadata: `{ "frame_index": <int>, "detections": [ ... ] }`
2. `image/jpeg` annotated frame bytes

Why multipart instead of SSE:
- Single inference per frame with tightly coupled metadata.
- Easy to extend (add extra JSON parts periodically for stats).

### Frontend

```bash
streamlit run src/frontend/app.py
```

The Streamlit client posts to `/video-detection` with a JSON body containing the `source` and parses the `multipart/mixed` stream (alternating JSON metadata and JPEG image parts). Set `PL8CATCH_BACKEND_ENDPOINT` in your `.env` (e.g. `http://localhost:800`).

## Docker
