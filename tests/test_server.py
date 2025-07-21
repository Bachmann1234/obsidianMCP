"""Tests for the MCP server module."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from obsidian_mcp.config import ServerConfig
from obsidian_mcp.server import ObsidianMCPServer


@pytest.fixture
def temp_vault():
    """Create a temporary vault for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        
        # Create some test notes
        (vault_path / "test1.md").write_text("""---
title: Test Note 1
tags: [python, testing]
---

# Test Note 1
This is a test note about Python.""")
        
        (vault_path / "test2.md").write_text("""# Test Note 2
This is another test note.""")
        
        yield vault_path


@pytest.fixture
def server_config(temp_vault):
    """Create a test server configuration."""
    return ServerConfig(
        vault_path=temp_vault,
        max_results=10,
        auto_rebuild_index=False,  # Don't auto-rebuild for tests
        watch_for_changes=False    # Don't watch for tests
    )


@pytest.fixture
def mock_server():
    """Create a mock MCP server."""
    server = Mock()
    server.list_tools = Mock()
    server.call_tool = Mock()
    return server


class TestObsidianMCPServer:
    """Test cases for ObsidianMCPServer."""
    
    def test_server_initialization(self, server_config):
        """Test server initialization."""
        with patch('obsidian_mcp.server.Server') as mock_server_class:
            mock_server = Mock()
            mock_server_class.return_value = mock_server
            
            server = ObsidianMCPServer(server_config)
            
            assert server.config == server_config
            assert server.parser is not None
            assert server.search_index is not None
            assert server.watcher is not None
            assert server.server == mock_server
            
            # Verify server was created with correct name
            mock_server_class.assert_called_once_with("obsidian-mcp-server")
    
    @pytest.mark.asyncio
    async def test_initialize_index(self, server_config):
        """Test index initialization."""
        with patch('obsidian_mcp.server.Server'):
            server = ObsidianMCPServer(server_config)
            
            # Mock the parser and search_index methods
            with patch.object(server.parser, 'discover_notes') as mock_discover, \
                 patch.object(server.parser, 'parse_note') as mock_parse, \
                 patch.object(server.parser, 'compute_backlinks') as mock_backlinks, \
                 patch.object(server.search_index, 'rebuild_index') as mock_rebuild:
                
                # Setup mocks
                mock_note = Mock()
                mock_discover.return_value = [Path("test.md")]
                mock_parse.return_value = mock_note
                
                await server._initialize_index()
                
                mock_discover.assert_called_once()
                mock_parse.assert_called_once_with(Path("test.md"))
                mock_backlinks.assert_called_once_with([mock_note])
                mock_rebuild.assert_called_once_with([mock_note])
    
    @pytest.mark.asyncio
    async def test_search_notes_basic(self, server_config):
        """Test basic search functionality."""
        with patch('obsidian_mcp.server.Server'):
            server = ObsidianMCPServer(server_config)
            
            # Mock search results
            mock_results = [
                {
                    'path': '/vault/test.md',
                    'title': 'Test Note',
                    'content': 'Test content',
                    'tags': ['python'],
                    'score': 0.95,
                    'highlights': {},
                    'created_date': '2024-01-01T00:00:00',
                    'modified_date': '2024-01-01T00:00:00'
                }
            ]
            
            with patch.object(server.search_index, 'search', return_value=mock_results):
                result = await server._search_notes({"query": "python"})
                
                assert len(result) == 1
                content = result[0].text
                assert "Found 1 results for 'python'" in content
                assert "Test Note" in content
                assert "Score: 0.95" in content
    
    @pytest.mark.asyncio
    async def test_search_notes_empty_query(self, server_config):
        """Test search with empty query."""
        with patch('obsidian_mcp.server.Server'):
            server = ObsidianMCPServer(server_config)
            
            result = await server._search_notes({"query": ""})
            
            assert len(result) == 1
            assert "Empty search query" in result[0].text
    
    @pytest.mark.asyncio
    async def test_search_notes_no_results(self, server_config):
        """Test search with no results."""
        with patch('obsidian_mcp.server.Server'):
            server = ObsidianMCPServer(server_config)
            
            with patch.object(server.search_index, 'search', return_value=[]):
                result = await server._search_notes({"query": "nonexistent"})
                
                assert len(result) == 1
                assert "No results found for query: nonexistent" in result[0].text
    
    @pytest.mark.asyncio
    async def test_search_notes_with_tags(self, server_config):
        """Test search with tag filtering."""
        with patch('obsidian_mcp.server.Server'):
            server = ObsidianMCPServer(server_config)
            
            mock_results = [{'path': '/test.md', 'title': 'Test', 'content': 'Content', 
                           'tags': ['python'], 'score': 0.9, 'highlights': {},
                           'created_date': None, 'modified_date': None}]
            
            with patch.object(server.search_index, 'search', return_value=mock_results) as mock_search:
                await server._search_notes({"query": "test", "tags": ["python"], "limit": 5})
                
                mock_search.assert_called_once_with(
                    query="test",
                    limit=5,
                    tags={"python"}
                )
    
    @pytest.mark.asyncio
    async def test_get_note_by_path(self, server_config):
        """Test getting a note by its path."""
        with patch('obsidian_mcp.server.Server'):
            server = ObsidianMCPServer(server_config)
            
            mock_result = {
                'path': str(server_config.vault_path / "test.md"),
                'title': 'Test Note',
                'content': 'Test content',
                'tags': ['python'],
                'wikilinks': [],
                'created_date': '2024-01-01T00:00:00',
                'modified_date': '2024-01-01T00:00:00'
            }
            
            # Mock the file existence check and search
            with patch.object(server.search_index, 'get_note_by_path', return_value=mock_result), \
                 patch('pathlib.Path.exists', return_value=True):
                
                result = await server._get_note({"identifier": "test.md"})
                
                assert len(result) == 1
                content = result[0].text
                assert "# Test Note" in content
                assert "**Path:**" in content
                assert "**Tags:** python" in content
                assert "Test content" in content
    
    @pytest.mark.asyncio
    async def test_get_note_not_found(self, server_config):
        """Test getting a non-existent note."""
        with patch('obsidian_mcp.server.Server'):
            server = ObsidianMCPServer(server_config)
            
            with patch.object(server.search_index, 'get_note_by_path', return_value=None), \
                 patch.object(server.search_index, 'search', return_value=[]):
                
                result = await server._get_note({"identifier": "nonexistent.md"})
                
                assert len(result) == 1
                assert "Note not found: nonexistent.md" in result[0].text
    
    @pytest.mark.asyncio
    async def test_get_note_by_title_search(self, server_config):
        """Test getting a note by title through search fallback."""
        with patch('obsidian_mcp.server.Server'):
            server = ObsidianMCPServer(server_config)
            
            mock_search_result = {
                'path': '/vault/test.md',
                'title': 'Test Note',
                'content': 'Content',
                'tags': [],
                'wikilinks': [],
                'created_date': None,
                'modified_date': None
            }
            
            with patch.object(server.search_index, 'get_note_by_path', return_value=None), \
                 patch.object(server.search_index, 'search', return_value=[mock_search_result]):
                
                result = await server._get_note({"identifier": "Test Note"})
                
                assert len(result) == 1
                assert "# Test Note" in result[0].text
    
    @pytest.mark.asyncio
    async def test_list_recent_notes(self, server_config):
        """Test listing recent notes."""
        with patch('obsidian_mcp.server.Server'):
            server = ObsidianMCPServer(server_config)
            
            mock_recent = [
                {
                    'path': '/vault/recent.md',
                    'title': 'Recent Note',
                    'content': 'Recent content',
                    'tags': ['recent'],
                    'modified_date': '2024-01-01T00:00:00'
                }
            ]
            
            with patch.object(server.search_index, 'get_recent_notes', return_value=mock_recent):
                result = await server._list_recent_notes({"limit": 5})
                
                assert len(result) == 1
                content = result[0].text
                assert "Recently modified notes (showing 1)" in content
                assert "Recent Note" in content
                assert "Modified: 2024-01-01T00:00:00" in content
    
    @pytest.mark.asyncio
    async def test_list_recent_notes_empty(self, server_config):
        """Test listing recent notes when none exist."""
        with patch('obsidian_mcp.server.Server'):
            server = ObsidianMCPServer(server_config)
            
            with patch.object(server.search_index, 'get_recent_notes', return_value=[]):
                result = await server._list_recent_notes({})
                
                assert len(result) == 1
                assert "No notes found" in result[0].text
    
    @pytest.mark.asyncio
    async def test_get_all_tags(self, server_config):
        """Test getting all tags."""
        with patch('obsidian_mcp.server.Server'):
            server = ObsidianMCPServer(server_config)
            
            mock_tags = ['python', 'testing', 'notes']
            
            with patch.object(server.search_index, 'list_all_tags', return_value=mock_tags):
                result = await server._get_all_tags({})
                
                assert len(result) == 1
                content = result[0].text
                assert "Found 3 tags:" in content
                assert "- python" in content
                assert "- testing" in content
                assert "- notes" in content
    
    @pytest.mark.asyncio
    async def test_get_all_tags_empty(self, server_config):
        """Test getting all tags when none exist."""
        with patch('obsidian_mcp.server.Server'):
            server = ObsidianMCPServer(server_config)
            
            with patch.object(server.search_index, 'list_all_tags', return_value=[]):
                result = await server._get_all_tags({})
                
                assert len(result) == 1
                assert "No tags found in the vault" in result[0].text
    
    @pytest.mark.asyncio
    async def test_search_by_tag(self, server_config):
        """Test searching by specific tags."""
        with patch('obsidian_mcp.server.Server'):
            server = ObsidianMCPServer(server_config)
            
            mock_results = [
                {
                    'path': '/vault/python.md',
                    'title': 'Python Notes',
                    'content': 'Python is great for...',
                    'tags': ['python', 'programming'],
                    'score': 0.9,
                    'highlights': {},
                    'created_date': None,
                    'modified_date': None
                }
            ]
            
            with patch.object(server.search_index, 'search', return_value=mock_results) as mock_search:
                result = await server._search_by_tag({"tags": ["python"], "limit": 10})
                
                assert len(result) == 1
                content = result[0].text
                assert "Found 1 notes with tags python" in content
                assert "Python Notes" in content
                
                # Verify search was called with correct tag query
                mock_search.assert_called_once_with(query="tags:python", limit=10)
    
    @pytest.mark.asyncio
    async def test_search_by_tag_multiple(self, server_config):
        """Test searching by multiple tags."""
        with patch('obsidian_mcp.server.Server'):
            server = ObsidianMCPServer(server_config)
            
            with patch.object(server.search_index, 'search', return_value=[]) as mock_search:
                # Don't specify limit, use the default (which is 20 but limited by config.max_results=10)
                await server._search_by_tag({"tags": ["python", "testing"]})
                
                # The limit should be min(20, config.max_results) = min(20, 10) = 10
                mock_search.assert_called_once_with(query="tags:python OR tags:testing", limit=10)
    
    @pytest.mark.asyncio
    async def test_search_by_tag_no_tags(self, server_config):
        """Test searching by tags with no tags provided."""
        with patch('obsidian_mcp.server.Server'):
            server = ObsidianMCPServer(server_config)
            
            result = await server._search_by_tag({"tags": []})
            
            assert len(result) == 1
            assert "No tags specified" in result[0].text
    
    @pytest.mark.asyncio
    async def test_get_vault_stats(self, server_config):
        """Test getting vault statistics."""
        with patch('obsidian_mcp.server.Server'):
            server = ObsidianMCPServer(server_config)
            
            mock_index_stats = {
                'doc_count': 42,
                'index_path': '/test/index',
                'field_names': ['title', 'content', 'tags']
            }
            
            mock_watcher_stats = {
                'enabled': True,
                'running': False,
                'files_updated': 5,
                'files_deleted': 2,
                'files_moved': 1
            }
            
            with patch.object(server.search_index, 'get_stats', return_value=mock_index_stats), \
                 patch.object(server.watcher, 'get_stats', return_value=mock_watcher_stats), \
                 patch.object(server.parser, 'discover_notes', return_value=['note1.md', 'note2.md']):
                
                result = await server._get_vault_stats({})
                
                assert len(result) == 1
                content = result[0].text
                assert "# Vault Statistics" in content
                assert "**Total markdown files:** 2" in content
                assert "**Indexed documents:** 42" in content
                assert "**Enabled:** True" in content
                assert "**Files updated:** 5" in content
    
    @pytest.mark.asyncio
    async def test_get_vault_stats_error(self, server_config):
        """Test getting vault statistics when an error occurs."""
        with patch('obsidian_mcp.server.Server'):
            server = ObsidianMCPServer(server_config)
            
            with patch.object(server.search_index, 'get_stats', side_effect=Exception("Index error")):
                result = await server._get_vault_stats({})
                
                assert len(result) == 1
                assert "Error retrieving stats: Index error" in result[0].text
    
    @pytest.mark.asyncio
    async def test_run_with_watcher(self, server_config):
        """Test running the server with file watcher."""
        with patch('obsidian_mcp.server.Server') as mock_server_class, \
             patch('obsidian_mcp.server.stdio_server') as mock_stdio_server:
            
            mock_server = Mock()
            mock_server_class.return_value = mock_server
            
            # Mock stdio_server as async context manager
            mock_streams = (Mock(), Mock())
            mock_stdio_server.return_value.__aenter__ = AsyncMock(return_value=mock_streams)
            mock_stdio_server.return_value.__aexit__ = AsyncMock(return_value=None)
            
            mock_server.run = AsyncMock()
            
            server = ObsidianMCPServer(server_config)
            
            with patch.object(server.watcher, 'start') as mock_start, \
                 patch.object(server.watcher, 'stop') as mock_stop:
                
                await server.run()
                
                mock_start.assert_called_once()
                mock_stop.assert_called_once()
                mock_server.run.assert_called_once_with(*mock_streams)


class TestServerMain:
    """Test cases for the main CLI function."""
    
    def test_main_with_vault_path(self, temp_vault):
        """Test main function with vault path argument."""
        with patch('obsidian_mcp.server.ObsidianMCPServer') as mock_server_class, \
             patch('asyncio.run') as mock_asyncio_run:
            
            mock_server = Mock()
            mock_server_class.return_value = mock_server
            
            from obsidian_mcp.server import main
            from click.testing import CliRunner
            
            runner = CliRunner()
            result = runner.invoke(main, ['--vault-path', str(temp_vault)])
            
            assert result.exit_code == 0
            mock_server_class.assert_called_once()
            mock_asyncio_run.assert_called_once()
    
    def test_main_with_env_config(self, temp_vault, monkeypatch):
        """Test main function using environment configuration."""
        monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(temp_vault))
        
        with patch('obsidian_mcp.server.ObsidianMCPServer') as mock_server_class, \
             patch('asyncio.run') as mock_asyncio_run:
            
            mock_server = Mock()
            mock_server_class.return_value = mock_server
            
            from obsidian_mcp.server import main
            from click.testing import CliRunner
            
            runner = CliRunner()
            result = runner.invoke(main, [])
            
            assert result.exit_code == 0
            mock_server_class.assert_called_once()
            mock_asyncio_run.assert_called_once()
    
    def test_main_with_all_options(self, temp_vault):
        """Test main function with all command line options."""
        custom_index = temp_vault / "custom_index"
        
        with patch('obsidian_mcp.server.ObsidianMCPServer') as mock_server_class, \
             patch('asyncio.run') as mock_asyncio_run:
            
            mock_server = Mock()
            mock_server_class.return_value = mock_server
            
            from obsidian_mcp.server import main
            from click.testing import CliRunner
            
            runner = CliRunner()
            result = runner.invoke(main, [
                '--vault-path', str(temp_vault),
                '--index-path', str(custom_index),
                '--max-results', '100'
            ])
            
            assert result.exit_code == 0
            
            # Verify ServerConfig was created with correct parameters
            call_args = mock_server_class.call_args[0][0]
            assert call_args.vault_path.resolve() == temp_vault.resolve()
            assert call_args.index_path.resolve() == custom_index.resolve()
            assert call_args.max_results == 100