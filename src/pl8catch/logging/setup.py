"""Logging configuration"""

import sys
from typing import Literal

from loguru import logger


def configure_logging(level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]) -> None:
    """Configure Loguru logging sinks and format."""
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        level=level,
    )
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        level="DEBUG",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    return None
