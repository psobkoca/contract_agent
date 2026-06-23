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

class RAGConfig(BaseModel):
    """Configuration settings for the Legal RAG Retrieval Engine."""
    chunk_size: int = Field(
        512,
        description="Text splitter token chunk size."
    )
    chunk_overlap: int = Field(
        64,
        description="Text splitter token chunk overlap."
    )
    top_k: int = Field(
        6,
        description="Number of candidate passages to retrieve in hybrid search before reranking."
    )

class AgentConfig(BaseModel):
    """Configuration settings for the Review Agent."""
    max_clauses_per_run: int = Field(
        10,
        description="Maximum number of clauses to analyze per contract run."
    )
    model: str = Field(
        "claude-sonnet-4-20250514",
        description="The target Anthropic model to call."
    )

class AppConfig(BaseSettings):
    """Application-wide settings loader."""
    parsing: ParsingConfig = ParsingConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    rag: RAGConfig = RAGConfig()
    agent: AgentConfig = AgentConfig()

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_prefix="APP_",
        extra="ignore"
    )

config = AppConfig()
