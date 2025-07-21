"""Configuration management for Obsidian MCP Server."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, validator


class ServerConfig(BaseModel):
    """Configuration for the Obsidian MCP Server."""
    
    vault_path: Path = Field(
        description="Path to the Obsidian vault directory"
    )
    index_path: Optional[Path] = Field(
        default=None,
        description="Path to store the search index (defaults to vault_path/.obsidian-mcp-index)"
    )
    max_results: int = Field(
        default=50,
        description="Maximum number of search results to return"
    )
    auto_rebuild_index: bool = Field(
        default=True,
        description="Whether to automatically rebuild index on startup if needed"
    )
    watch_for_changes: bool = Field(
        default=True,
        description="Whether to watch for file changes and update index incrementally"
    )
    include_content_in_search: bool = Field(
        default=True,
        description="Whether to include full note content in search results"
    )
    
    @validator("vault_path")
    def vault_path_must_exist(cls, v: Path) -> Path:
        """Validate that vault path exists and is a directory."""
        if not v.exists():
            raise ValueError(f"Vault path does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Vault path is not a directory: {v}")
        return v.resolve()
    
    @validator("index_path", always=True)
    def set_default_index_path(cls, v: Optional[Path], values: dict) -> Path:
        """Set default index path if not provided."""
        if v is None:
            vault_path = values.get("vault_path")
            if vault_path:
                return vault_path / ".obsidian-mcp-index"
        return v.resolve() if v else Path.cwd() / ".obsidian-mcp-index"


def load_config_from_env() -> ServerConfig:
    """Load configuration from environment variables."""
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
    if not vault_path:
        raise ValueError(
            "OBSIDIAN_VAULT_PATH environment variable must be set"
        )
    
    config_data = {
        "vault_path": Path(vault_path),
    }
    
    # Optional environment variables
    if index_path := os.getenv("OBSIDIAN_INDEX_PATH"):
        config_data["index_path"] = Path(index_path)
    
    if max_results := os.getenv("OBSIDIAN_MAX_RESULTS"):
        config_data["max_results"] = int(max_results)
    
    if auto_rebuild := os.getenv("OBSIDIAN_AUTO_REBUILD_INDEX"):
        config_data["auto_rebuild_index"] = auto_rebuild.lower() == "true"
    
    if watch_changes := os.getenv("OBSIDIAN_WATCH_CHANGES"):
        config_data["watch_for_changes"] = watch_changes.lower() == "true"
    
    if include_content := os.getenv("OBSIDIAN_INCLUDE_CONTENT"):
        config_data["include_content_in_search"] = include_content.lower() == "true"
    
    return ServerConfig(**config_data)