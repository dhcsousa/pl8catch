"""System / configuration endpoints."""

from importlib.metadata import version

from fastapi import APIRouter, Request

from pl8catch.config import AppConfig

router = APIRouter()


@router.get("/config", summary="Current server configuration")
async def current_config(
    request: Request,
) -> dict:
    """Return the effective runtime configuration (non-sensitive)."""
    config: AppConfig = request.app.state.config
    return {
        "version": version("pl8catch"),
        "server": config.server.model_dump(),
        "models": config.models.model_dump(),
        "license_plate_ocr": config.license_plate_ocr.model_dump(),
    }
