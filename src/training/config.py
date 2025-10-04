from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from pl8catch.config.app_config import BaseFileConfig


class Environment(BaseSettings):
    """
    A class representing the environment variables and settings for the training step.

    Attributes:
        LOG_LEVEL (Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]): The level of logging for the application.
        TRAINING_CONFIG_FILE_PATH (Path): The path to the training configuration file.
    """

    ROOT_DIR: Path = Path(__file__).resolve().parents[2]

    LOG_LEVEL: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
    TRAINING_CONFIG_FILE_PATH: Path = ROOT_DIR / "configs" / "train.yaml"

    ROBOFLOW_API_KEY: SecretStr | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


class TrainConfig(BaseFileConfig):
    """Configuration parameters for the YOLO training session, sourced from the train YAML config file."""

    ROOT_DIR: Path = Path(__file__).resolve().parents[2]

    datasets_dir: Path = ROOT_DIR / "downloaded_dataset"  # Directory registered with Ultralytics for dataset lookups
    data_config_path: Path = datasets_dir / "data.yaml"  # Path to the YOLO dataset configuration file
    weights_path: Path | str = (
        "yolo12s.pt"  # Checkpoint used to initialise the YOLO model before training, it might be a string like "yolo12s.pt" and the model will be downloaded
    )
    epochs: int  # Number of epochs to train for
    image_size: int  # Image size for training
    batch_size: int | None = None  # Optional batch size override
    device: str  # Device identifier
    project_name: str = "yolo_runs"  # Directory for training artefacts
    run_name: str | None = None  # Optional run identifier
    resume: bool = False  # Resume training from last checkpoint
    patience: int | None = None  # Optional early stopping patience
    seed: int | None = None  # Optional random seed
    profile: bool = False  # Enable Ultralytics profiler
    fraction: float | None = None  # Optional dataset fraction (e.g., 0.1 to use 10% of the data)
    extra_train_args: dict[str, object] = Field(default_factory=dict)  # Additional keyword arguments for YOLO.train

    # MLflow tracking settings
    mlflow_enabled: bool = True  # Toggle MLflow tracking
    mlflow_tracking_uri: Path = ROOT_DIR / "mlflow"  # MLflow tracking URI
    mlflow_experiment_name: str = "pl8catch"  # MLflow experiment name
    mlflow_run_name: str  # Explicit MLflow run name

    # Roboflow dataset metadata
    roboflow_workspace: str = "roboflow-universe-projects"
    roboflow_project: str = "license-plate-recognition-rxg4e"
    roboflow_version: int = 11
    roboflow_export_format: str = "yolov12"
