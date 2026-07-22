"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration with safe local defaults."""

    app_name: str = "SupportIQ"
    environment: str = "development"
    log_level: str = "INFO"
    database_url: str = "sqlite:///./data/supportiq.db"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8080"]
    model_dir: Path = Path("./models")
    data_dir: Path = Path("./data")
    demo_mode: bool = True
    max_ticket_length: int = 10_000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""

    return Settings()


settings = get_settings()
