from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any

from fastapi import FastAPI
from loguru import logger

from pl8catch.data_model import AppConfig
from pl8catch.environment import Environment


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, Any]:
    env = Environment()
    app.state.env = env

    # Application configuration
    logger.info(f"Using configuration file at {str(env.CONFIG_FILE_PATH)}.")
    config: AppConfig = AppConfig.from_file(env.CONFIG_FILE_PATH)
    app.state.tool_config = config

    logger.info("Starting pl8catch backend server...")

    yield

    logger.info("Stopping pl8catch backend server...")
