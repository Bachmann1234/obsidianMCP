"""Tests for the configuration module."""

import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from obsidian_mcp.config import ServerConfig, load_config_from_env


@pytest.fixture
def temp_vault():
    """Create a temporary vault directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        yield vault_path


def test_server_config_creation(temp_vault):
    """Test creating ServerConfig with valid parameters."""
    config = ServerConfig(vault_path=temp_vault)

    assert config.vault_path == temp_vault.resolve()
    assert config.index_path == (temp_vault / ".obsidian-mcp-index").resolve()
    assert config.max_results == 50
    assert config.auto_rebuild_index is True
    assert config.watch_for_changes is True
    assert config.include_content_in_search is True


def test_server_config_with_custom_index_path(temp_vault):
    """Test ServerConfig with custom index path."""
    custom_index = temp_vault / "custom_index"
    config = ServerConfig(vault_path=temp_vault, index_path=custom_index)

    assert config.index_path == custom_index.resolve()


def test_server_config_with_custom_settings(temp_vault):
    """Test ServerConfig with custom settings."""
    config = ServerConfig(
        vault_path=temp_vault,
        max_results=100,
        auto_rebuild_index=False,
        watch_for_changes=False,
        include_content_in_search=False,
    )

    assert config.max_results == 100
    assert config.auto_rebuild_index is False
    assert config.watch_for_changes is False
    assert config.include_content_in_search is False


def test_server_config_validates_vault_path():
    """Test that ServerConfig validates vault path exists."""
    nonexistent_path = Path("/nonexistent/vault")

    with pytest.raises(ValidationError) as exc_info:
        ServerConfig(vault_path=nonexistent_path)

    assert "Vault path does not exist" in str(exc_info.value)


def test_server_config_validates_vault_is_directory(temp_vault):
    """Test that ServerConfig validates vault path is a directory."""
    # Create a file instead of directory
    file_path = temp_vault / "not_a_directory.txt"
    file_path.write_text("content")

    with pytest.raises(ValidationError) as exc_info:
        ServerConfig(vault_path=file_path)

    assert "Vault path is not a directory" in str(exc_info.value)


def test_load_config_from_env_basic(temp_vault, monkeypatch):
    """Test loading configuration from environment variables."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(temp_vault))

    config = load_config_from_env()

    assert config.vault_path == temp_vault.resolve()
    assert config.max_results == 50  # default


def test_load_config_from_env_all_options(temp_vault, monkeypatch):
    """Test loading all configuration options from environment."""
    custom_index = temp_vault / "custom_index"

    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(temp_vault))
    monkeypatch.setenv("OBSIDIAN_INDEX_PATH", str(custom_index))
    monkeypatch.setenv("OBSIDIAN_MAX_RESULTS", "75")
    monkeypatch.setenv("OBSIDIAN_AUTO_REBUILD_INDEX", "false")
    monkeypatch.setenv("OBSIDIAN_WATCH_CHANGES", "false")
    monkeypatch.setenv("OBSIDIAN_INCLUDE_CONTENT", "false")

    config = load_config_from_env()

    assert config.vault_path == temp_vault.resolve()
    assert config.index_path == custom_index.resolve()
    assert config.max_results == 75
    assert config.auto_rebuild_index is False
    assert config.watch_for_changes is False
    assert config.include_content_in_search is False


def test_load_config_from_env_missing_vault_path(monkeypatch):
    """Test that missing OBSIDIAN_VAULT_PATH raises an error."""
    # Ensure the environment variable is not set
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)

    with pytest.raises(ValueError) as exc_info:
        load_config_from_env()

    assert "OBSIDIAN_VAULT_PATH environment variable must be set" in str(exc_info.value)


def test_load_config_from_env_boolean_parsing(temp_vault, monkeypatch):
    """Test boolean environment variable parsing."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(temp_vault))

    # Test various boolean representations
    test_cases = [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("yes", False),  # Only "true" should be True
        ("1", False),  # Only "true" should be True
    ]

    for env_value, expected in test_cases:
        monkeypatch.setenv("OBSIDIAN_AUTO_REBUILD_INDEX", env_value)
        config = load_config_from_env()
        assert config.auto_rebuild_index is expected


def test_default_index_path_validator(temp_vault):
    """Test the default index path validator."""
    config = ServerConfig(vault_path=temp_vault, index_path=None)

    # When index_path is None, it should default to vault_path/.obsidian-mcp-index
    expected_path = temp_vault / ".obsidian-mcp-index"
    assert config.index_path == expected_path.resolve()


def test_config_with_relative_paths():
    """Test that relative paths are resolved to absolute paths."""
    # Use current directory which should exist
    current_dir = Path.cwd()

    config = ServerConfig(vault_path=current_dir)

    # Paths should be resolved to absolute
    assert config.vault_path.is_absolute()
    assert config.index_path.is_absolute()


def test_load_config_partial_env_vars(temp_vault, monkeypatch):
    """Test loading config with only some environment variables set."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(temp_vault))
    monkeypatch.setenv("OBSIDIAN_MAX_RESULTS", "25")
    # Leave other variables unset

    config = load_config_from_env()

    assert config.vault_path == temp_vault.resolve()
    assert config.max_results == 25
    # Other values should use defaults
    assert config.auto_rebuild_index is True
    assert config.watch_for_changes is True
    assert config.include_content_in_search is True
