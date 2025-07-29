"""Configuration management for Obsidian MCP Server."""

import os
from pathlib import Path
from typing import Optional, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class ServerConfig(BaseModel):
    """Configuration for the Obsidian MCP Server."""

    vault_path: Path = Field(description="Path to the Obsidian vault directory")
    index_path: Optional[Path] = Field(
        default=None,
        description="Path to store the search index (defaults to vault_path/.obsidian-mcp-index)",
        validate_default=True,
    )
    max_results: int = Field(
        default=50, description="Maximum number of search results to return"
    )
    auto_rebuild_index: bool = Field(
        default=True,
        description="Whether to automatically rebuild index on startup if needed",
    )
    incremental_update: bool = Field(
        default=True,
        description="Whether to use incremental updates when possible instead of full rebuild",
    )
    watch_for_changes: bool = Field(
        default=True,
        description="Whether to watch for file changes and update index incrementally",
    )
    include_content_in_search: bool = Field(
        default=True,
        description="Whether to include full note content in search results",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformers model name for generating embeddings",
    )
    vector_index_path: Optional[Path] = Field(
        default=None,
        description="Path to store the vector index (defaults to vault_path/.obsidian-vector-index)",
        validate_default=True,
    )
    hybrid_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for combining text and vector search (0.0 = text only, 1.0 = vector only)",
    )
    use_polling_observer: bool = Field(
        default=False,
        description="Use polling-based file watcher instead of native OS events (better for Docker/network drives)",
    )

    @field_validator("vault_path")
    @classmethod
    def vault_path_must_exist(cls, v: Path) -> Path:
        """Validate that vault path exists and is a directory."""
        if not v.exists():
            raise ValueError(f"Vault path does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Vault path is not a directory: {v}")
        return v.resolve()

    @model_validator(mode="after")
    def set_default_paths(self) -> Self:
        """Set default index and vector index paths if not provided."""
        if self.index_path is None:
            object.__setattr__(
                self, "index_path", self.vault_path / ".obsidian-mcp-index"
            )
        else:
            object.__setattr__(self, "index_path", self.index_path.resolve())

        if self.vector_index_path is None:
            object.__setattr__(
                self, "vector_index_path", self.vault_path / ".obsidian-vector-index"
            )
        else:
            object.__setattr__(
                self, "vector_index_path", self.vector_index_path.resolve()
            )

        return self


def load_config_from_env() -> ServerConfig:
    """Load configuration from environment variables."""
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
    if not vault_path:
        raise ValueError("OBSIDIAN_VAULT_PATH environment variable must be set")

    config_kwargs: dict[str, object] = {
        "vault_path": Path(vault_path),
    }

    # Optional environment variables
    if index_path := os.getenv("OBSIDIAN_INDEX_PATH"):
        config_kwargs["index_path"] = Path(index_path)

    if max_results := os.getenv("OBSIDIAN_MAX_RESULTS"):
        config_kwargs["max_results"] = int(max_results)

    if auto_rebuild := os.getenv("OBSIDIAN_AUTO_REBUILD_INDEX"):
        config_kwargs["auto_rebuild_index"] = auto_rebuild.lower() == "true"

    if watch_changes := os.getenv("OBSIDIAN_WATCH_CHANGES"):
        config_kwargs["watch_for_changes"] = watch_changes.lower() == "true"

    if include_content := os.getenv("OBSIDIAN_INCLUDE_CONTENT"):
        config_kwargs["include_content_in_search"] = include_content.lower() == "true"

    if incremental := os.getenv("OBSIDIAN_INCREMENTAL_UPDATE"):
        config_kwargs["incremental_update"] = incremental.lower() == "true"

    if embedding_model := os.getenv("OBSIDIAN_EMBEDDING_MODEL"):
        config_kwargs["embedding_model"] = embedding_model

    if vector_index_path := os.getenv("OBSIDIAN_VECTOR_INDEX_PATH"):
        config_kwargs["vector_index_path"] = Path(vector_index_path)

    if hybrid_alpha := os.getenv("OBSIDIAN_HYBRID_ALPHA"):
        config_kwargs["hybrid_alpha"] = float(hybrid_alpha)

    if use_polling := os.getenv("OBSIDIAN_USE_POLLING_OBSERVER"):
        config_kwargs["use_polling_observer"] = use_polling.lower() == "true"

    return ServerConfig(**config_kwargs)  # type: ignore[arg-type]
