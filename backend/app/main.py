"""SupportIQ FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api import router
from backend.app.config import settings
from backend.app.database import SessionLocal, init_db
from backend.app.logging import configure_logging
from backend.app.ml_service import ModelService
from backend.app.schemas import HealthResponse
from backend.app.seed import seed_if_empty

configure_logging(settings.log_level)
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    app.state.model_service = ModelService()
    with SessionLocal() as session:
        seeded = seed_if_empty(session, app.state.model_service)
    logger.info("application_started", demo_mode=app.state.model_service.demo_mode, seeded=seeded)
    yield
    logger.info("application_stopped")


def create_app(*, initialize_database: bool = True, model_dir: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def inference_only_lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Initialize inference only, allowing isolated API tests to own persistence."""

        app.state.model_service = ModelService(model_dir)
        yield

    application = FastAPI(
        title="SupportIQ API",
        version="1.0.0",
        description="AI-assisted customer support ticket intelligence.",
        lifespan=lifespan if initialize_database else inference_only_lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next: object):
        request_id = request.headers.get("x-request-id", str(uuid4()))
        started = perf_counter()
        try:
            response = await call_next(request)  # type: ignore[operator]
        except Exception:
            logger.exception(
                "unhandled_request_error", request_id=request_id, path=request.url.path
            )
            raise
        response.headers["x-request-id"] = request_id
        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round((perf_counter() - started) * 1000, 2),
        )
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder({"error": "validation_error", "detail": exc.errors()}),
        )

    @application.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("internal_server_error", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "detail": "An unexpected error occurred"},
        )

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request) -> HealthResponse:
        service: ModelService = request.app.state.model_service
        return HealthResponse(
            status="ok",
            service=settings.app_name,
            demo_mode=service.demo_mode,
            model_status=service.status,
        )

    application.include_router(router)
    return application


app = create_app()
