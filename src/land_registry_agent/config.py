"""Environment-driven application configuration."""

from pathlib import Path
from typing import Self

from pydantic import Field, HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: SecretStr | None = None
    openai_intent_model: str = "gpt-4o-mini"
    openai_draft_model: str = "gpt-4o-mini"

    land_registry_sparql_url: HttpUrl = HttpUrl(
        "https://landregistry.data.gov.uk/landregistry/query"
    )
    hpi_region_base_url: HttpUrl = HttpUrl(
        "https://landregistry.data.gov.uk/data/hpi/region"
    )
    http_timeout_seconds: float = Field(default=60.0, gt=0, le=120)
    http_max_retries: int = Field(default=2, ge=0, le=5)
    query_page_size: int = Field(default=1_000, ge=100, le=10_000)
    query_max_pages: int = Field(default=20, ge=1, le=100)
    cache_ttl_seconds: int = Field(default=3_600, ge=0)

    checkpoint_db_path: Path = Path("data/checkpoints.sqlite3")
    reports_db_path: Path = Path("data/reports.sqlite3")
    cache_directory: Path = Path("data/cache")

    minimum_local_transactions: int = Field(default=10, ge=1)
    minimum_street_transactions: int = Field(default=3, ge=1)
    maximum_corrective_redrafts: int = Field(default=1, ge=0, le=1)

    demo_user_id: str = "demo-user-001"
    demo_user_display_name: str = "Demo User"

    @field_validator(
        "openai_intent_model",
        "openai_draft_model",
        "demo_user_id",
        "demo_user_display_name",
    )
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        """Reject configuration values containing only whitespace."""

        value = value.strip()
        if not value:
            raise ValueError("Configuration value must not be blank.")
        return value

    @model_validator(mode="after")
    def require_distinct_databases(self) -> Self:
        """Keep workflow checkpoints separate from approved reports."""

        if self.checkpoint_db_path == self.reports_db_path:
            raise ValueError(
                "Checkpoint and approved-report databases must use different paths."
            )
        return self

    @property
    def openai_enabled(self) -> bool:
        """Indicate whether language-model services can be used."""

        return self.openai_api_key is not None
