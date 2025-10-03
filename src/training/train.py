"""Train the Pl8Catch YOLO model using the configuration file in ``configs/train.yaml``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger
from ultralytics import YOLO, settings
from roboflow import Roboflow

from training.config import Environment, TrainConfig


def configure_logging(level: str) -> None:
    """Configure Loguru with the provided log level."""

    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True)


def validate_paths(config: TrainConfig) -> tuple[Path | str, Path]:
    """Validate and resolve the weights and dataset configuration paths."""

    if isinstance(config.weights_path, Path) and not config.weights_path.exists():
        logger.warning(f"YOLO weights not found at {config.weights_path}, they will be downloaded if possible")

    if not config.data_config_path.exists():
        raise FileNotFoundError(
            "Dataset configuration not found. Expected YOLO data.yaml at " f"{config.data_config_path}"
        )

    return config.weights_path, config.data_config_path


def download_roboflow_dataset(env: Environment, config: TrainConfig) -> None:
    """Download and extract the dataset from Roboflow using the provided configuration."""

    if config.data_config_path.exists():
        logger.info(
            "Dataset configuration already present at {}. Skipping Roboflow download.",
            config.data_config_path,
        )
        return

    if env.ROBOFLOW_API_KEY is None:
        logger.error("Roboflow API key not set, skipping dataset download")
        return

    rf = Roboflow(api_key=env.ROBOFLOW_API_KEY.get_secret_value())
    workspace = rf.workspace(config.roboflow_workspace)
    project = workspace.project(config.roboflow_project)
    version = project.version(config.roboflow_version)

    logger.info(
        "Downloading dataset from Roboflow: workspace='{}', project='{}', version={}",
        config.roboflow_workspace,
        config.roboflow_project,
        config.roboflow_version,
    )
    dataset = version.download(config.roboflow_export_format, location=str(config.datasets_dir))

    if not config.data_config_path.exists():
        raise FileNotFoundError(
            "Dataset configuration not found after Roboflow download. "
            f"Expected YOLO data.yaml at {config.data_config_path}"
        )

    logger.info("Roboflow dataset downloaded and extracted to {}", dataset.location)


def configure_mlflow(config: TrainConfig) -> str | None:
    """Enable MLflow tracking for Ultralytics and return the resolved run name."""

    settings.update({"mlflow": config.mlflow_enabled})
    if not config.mlflow_enabled:
        logger.debug("MLflow tracking disabled via configuration")
        return config.run_name or "license_plate_training"

    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", config.mlflow_experiment_name)
    os.environ.setdefault("MLFLOW_RUN", config.mlflow_run_name)
    os.environ.setdefault("MLFLOW_TRACKING_URI", str(config.mlflow_tracking_uri))

    logger.debug(
        "MLflow configured: tracking_uri='{}', experiment='{}', run='{}'",
        os.environ["MLFLOW_TRACKING_URI"],
        os.environ["MLFLOW_EXPERIMENT_NAME"],
        os.environ["MLFLOW_RUN"],
    )

    return None


def build_train_arguments(config: TrainConfig, run_name: str, data_config_path: Path) -> dict[str, object]:
    """Create the keyword arguments passed into ``YOLO.train``."""

    train_args: dict[str, object] = {
        "data": str(data_config_path),
        "epochs": config.epochs,
        "imgsz": config.image_size,
        "device": config.device,
        "project": config.project_name,
        "name": run_name,
        "resume": config.resume,
        "profile": config.profile,
    }

    if config.batch_size is not None:
        train_args["batch"] = config.batch_size
    if config.patience is not None:
        train_args["patience"] = config.patience
    if config.seed is not None:
        train_args["seed"] = config.seed
    if config.fraction is not None:
        train_args["fraction"] = config.fraction
    if config.extra_train_args:
        train_args.update(config.extra_train_args)

    return train_args


def main() -> None:
    """Run a YOLO training session described by ``TrainConfig``."""

    env = Environment()
    config = TrainConfig.from_file(env.TRAINING_CONFIG_FILE_PATH)

    configure_logging(env.LOG_LEVEL)

    download_roboflow_dataset(env, config)

    weights_path, data_config_path = validate_paths(config)

    settings.update({"datasets_dir": str(config.datasets_dir)})

    configure_mlflow(config)

    logger.info("Starting YOLO training")
    logger.info("  Weights: {}", weights_path)
    logger.info("  Data config: {}", data_config_path)
    logger.info("  Device: {}", config.device)

    model = YOLO(weights_path)
    train_args = build_train_arguments(config, config.mlflow_run_name, data_config_path)

    logger.info("YOLO.train arguments: {}", train_args)
    model.train(**train_args)
    logger.info("YOLO training complete")


if __name__ == "__main__":
    main()
