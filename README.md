# Pl8Catch: The License Plate Tracker

Pl8Catch is a comprehensive license plate recognition system designed to detect vehicles, extract license plate information, and provide a user-friendly interface for tracking and managing license plates.

## Features

- **YOLO Integration**: Utilizes YOLOv9 for efficient and accurate vehicle and license plate detection.
- **OCR (Optical Character Recognition)**: Employs OCR techniques to extract text from license plates.
- **FastAPI Backend**: Wraps the functionality into API using FastAPI for seamless integration into other applications.
- **Streamlit Frontend**: Provides a frontend interface built with Streamlit for easy access and interaction. The frontend is just to demonstrate the API capabilities and is not intended for production use.
- **MLflow Tracking**: Integrates MLflow for tracking experiments, metrics, and models during training.

## Training with MLflow

Ultralytics has native MLflow support, so you can keep track of metrics, parameters, and artifacts during training.

1. Start a local MLflow server (or point `MLFLOW_TRACKING_URI` to an existing server):

	```bash
	mlflow server
	```

2. Kick off training with sensible defaults and MLflow logging enabled:

	```bash
	python src/training/train.py
	```

Check the configuration file at `configs/training.yaml`/`TrainConfig` class to customize the training process by modifying parameters such as epochs, batch size, image size, and more.

**Heads up:** The script looks for the dataset at `downloaded_dataset/data.yaml`. If it cannot find the file it will attempt to download the dataset from Roboflow. Provide your own API key via `ROBOFLOW_API_KEY` environment variable. For more information visit [Roboflow](https://roboflow.com/).

You can also override `ROBOFLOW_WORKSPACE`, `ROBOFLOW_PROJECT`, `ROBOFLOW_VERSION`, and `ROBOFLOW_EXPORT_FORMAT` to point at a different dataset. Look at the config file or the `TrainConfig` class for the defaults.

## Dataset

The dataset used for training the model can be found [here](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4).

## Execution

### Backend

```bash
python src/pl8catch/app.py
```

### Frontend

```bash
streamlit run src/frontend/app.py
```

## Docker
