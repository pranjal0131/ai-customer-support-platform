"""SQLAlchemy engine and request-scoped session dependencies."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.config import settings


class Base(DeclarativeBase):
    """Declarative model base."""


connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and always close it after the request."""

    with SessionLocal() as session:
        yield session


def init_db() -> None:
    """Create tables for local use; production can replace this with migrations."""

    from backend.app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
