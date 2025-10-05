from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(BaseSettings):
    """
    A class representing the environment variables and settings for the application.

    Attributes:
        ROOT_DIR (Path): The root directory of the project.
        LOG_LEVEL (Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]): The level of logging for the application. Defaults to 'INFO'.
        CONFIG_FILE_PATH (str): The path to the service definition file. Defaults to 'configs/backend.yaml' under ROOT_DIR.
        MODEL_DIR (Path): The directory where model files are stored. Defaults to 'models' under ROOT_DIR.
    """

    # Number of parent directories to traverse to reach the project root.
    ROOT_DIR: Path = Path(__file__).resolve().parents[3]

    LOG_LEVEL: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    CONFIG_FILE_PATH: Path = ROOT_DIR / "configs" / "backend.yaml"
    MODEL_DIR: Path = ROOT_DIR / "models"  # Model runtime settings (download-on-start). Actual URLs in AppConfig.

    model_config = SettingsConfigDict(env_file=".env", extra="allow")
