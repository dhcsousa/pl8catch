import sys
from loguru import logger


def configure_logging(level: str) -> None:
    """Configure Loguru with the provided log level."""

    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True)
