from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(BaseSettings):
    """
    A class representing the environment variables and settings for the application.

    Attributes:
        LOG_LEVEL (Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]): The level of logging for the application.
        CONFIG_FILE_PATH (str): The path to the service definition file.
    """

    LOG_LEVEL: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
    CONFIG_FILE_PATH: Path

    model_config = SettingsConfigDict(env_file=".env", extra="allow")
