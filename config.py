from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class ParsingConfig(BaseModel):
    """Configuration settings for the contract text parser."""
    min_clause_chars: int = Field(
        20, 
        description="Minimum character length for a segment to be classified as a valid clause."
    )

class AppConfig(BaseSettings):
    """Application-wide settings loader."""
    parsing: ParsingConfig = ParsingConfig()

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_prefix="APP_",
        extra="ignore"
    )

config = AppConfig()
