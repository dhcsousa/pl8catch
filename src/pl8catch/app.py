"""FastAPI backend of pl8catch"""

from importlib.metadata import version

import uvicorn
from fastapi import FastAPI

from pl8catch.api import health_router, inference_router, system_router
from pl8catch.config import AppConfig, Environment
from pl8catch.lifespan import lifespan
from pl8catch.logging import configure_logging

env = Environment()
config = AppConfig.from_file(env.CONFIG_FILE_PATH)
configure_logging(env.LOG_LEVEL)

app = FastAPI(
    title="Pl8Catch",
    version=version("pl8catch"),
    summary="A API for vehicle detection and license plate recognition.",
    lifespan=lifespan,
    contact={
        "name": "Daniel Sousa",
        "email": "danielsoussa@gmail.com",
    },
)

# General / system routes
app.include_router(health_router, tags=["system"])
app.include_router(system_router, tags=["system"])

# Inference routes
app.include_router(inference_router, tags=["inference"])

if __name__ == "__main__":
    uvicorn.run(app, port=config.server.port, host=config.server.host)
