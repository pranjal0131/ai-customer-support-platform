"""FastAPI dependencies shared by API routers."""

from fastapi import Request

from backend.app.ml_service import ModelService


def get_model_service(request: Request) -> ModelService:
    """Return the model service initialized during application startup."""

    return request.app.state.model_service
