from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from typing import List

class ParsingConfig(BaseModel):
    """Configuration settings for the contract text parser."""
    min_clause_chars: int = Field(
        20, 
        description="Minimum character length for a segment to be classified as a valid clause."
    )

class ClassifierConfig(BaseModel):
    """Configuration settings for the clause classifier."""
    confidence_threshold: float = Field(
        0.60,
        description="Confidence threshold below which zero-shot NLI fallback is triggered."
    )
    high_risk_types: List[str] = Field(
        ["Liability", "Indemnification", "IP"],
        description="Clause types flagged as high risk if confidence is high."
    )

class AppConfig(BaseSettings):
    """Application-wide settings loader."""
    parsing: ParsingConfig = ParsingConfig()
    classifier: ClassifierConfig = ClassifierConfig()

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_prefix="APP_",
        extra="ignore"
    )

config = AppConfig()
