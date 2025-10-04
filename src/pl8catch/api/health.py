"""Health related API routes."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

Health = Literal["healthy", "unhealthy"]


class HealthCheck(BaseModel):
    """Response model returned by the health check endpoint.

    Parameters
    ----------
    status : Literal["healthy", "unhealthy"]
        Health status of the service.
    """

    status: Health = "healthy"


@router.get("/health", summary="Health check")
async def get_health() -> HealthCheck:  # pragma: no cover
    """Return service health status."""
    return HealthCheck()
