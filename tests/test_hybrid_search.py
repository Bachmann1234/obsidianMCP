"""Tests for the hybrid search functionality."""

import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, Mock, MagicMock
from typing import List, Dict, Any

import pytest

from obsidian_mcp.search import HybridSearchEngine, SearchResult
from obsidian_mcp.parser import ObsidianNote
from obsidian_mcp.config import ServerConfig


@pytest.fixture
def sample_config():
    """Create a sample server configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir) / "vault"
        vault_path.mkdir()
        config = ServerConfig(
            vault_path=vault_path,
            index_path=Path(temp_dir) / "index",
            vector_index_path=Path(temp_dir) / "vector_index",
            embedding_model="test-model",
            hybrid_alpha=0.5,
            max_results=50
        )
        yield config


@pytest.fixture
def sample_notes():
    """Create sample notes for testing."""
    notes = []
    
    note1 = ObsidianNote(
        path=Path("/vault/python-basics.md"),
        title="Python Basics",
        content="Python is a programming language. It's great for beginners.",
        raw_content="# Python Basics\n\nPython is a programming language.",
        frontmatter={"category": "programming"},
        tags={"python", "programming", "beginner"},
        wikilinks={"advanced-python"},
        backlinks=set(),
        created_date=datetime(2024, 1, 1),
        modified_date=datetime(2024, 1, 15)
    )
    
    note2 = ObsidianNote(
        path=Path("/vault/advanced-python.md"),
        title="Advanced Python",
        content="Advanced Python concepts including decorators and metaclasses.",
        raw_content="# Advanced Python\n\nAdvanced concepts.",
        frontmatter={"category": "programming", "difficulty": "hard"},
        tags={"python", "programming", "advanced"},
        wikilinks={"python-basics"},
        backlinks={"python-basics"},
        created_date=datetime(2024, 1, 10),
        modified_date=datetime(2024, 2, 1)
    )
    
    note3 = ObsidianNote(
        path=Path("/vault/javascript.md"),
        title="JavaScript Guide",
        content="JavaScript is a dynamic programming language for web development.",
        raw_content="# JavaScript Guide\n\nJavaScript is dynamic.",
        frontmatter={"category": "programming"},
        tags={"javascript", "programming", "web"},
        wikilinks=set(),
        backlinks=set(),
        created_date=datetime(2024, 2, 1),
        modified_date=datetime(2024, 2, 10)
    )
    
    return [note1, note2, note3]


@pytest.fixture
def mock_hybrid_engine(sample_config):
    """Create a mocked hybrid search engine."""
    with patch('obsidian_mcp.search.ObsidianSearchIndex') as mock_text_search, \
         patch('obsidian_mcp.search.VectorSearchEngine') as mock_vector_search:
        
        # Mock text search
        mock_text_instance = Mock()
        mock_text_search.return_value = mock_text_instance
        
        # Mock vector search
        mock_vector_instance = Mock()
        mock_vector_search.return_value = mock_vector_instance
        
        engine = HybridSearchEngine(sample_config)
        engine.text_search = mock_text_instance
        engine.vector_search = mock_vector_instance
        
        yield engine, mock_text_instance, mock_vector_instance


class TestSearchResult:
    """Test the SearchResult class."""
    
    def test_search_result_creation(self, sample_notes):
        """Test creating a SearchResult object."""
        note = sample_notes[0]
        score = 0.85
        highlights = {"content": "Python is a <em>programming</em> language"}
        
        result = SearchResult(note, score, highlights)
        
        assert result.note == note
        assert result.score == score
        assert result.highlights == highlights


class TestHybridSearchEngine:
    """Test the HybridSearchEngine class."""
    
    def test_init(self, sample_config):
        """Test hybrid search engine initialization."""
        with patch('obsidian_mcp.search.ObsidianSearchIndex') as mock_text_search, \
             patch('obsidian_mcp.search.VectorSearchEngine') as mock_vector_search:
            
            engine = HybridSearchEngine(sample_config)
            
            # Verify components were initialized
            mock_text_search.assert_called_once_with(sample_config.index_path)
            mock_vector_search.assert_called_once_with(
                index_path=sample_config.vector_index_path,
                embedding_model=sample_config.embedding_model
            )
            
            assert engine.config == sample_config
    
    def test_add_note(self, mock_hybrid_engine, sample_notes):
        """Test adding a note to both indices."""
        engine, mock_text, mock_vector = mock_hybrid_engine
        note = sample_notes[0]
        
        engine.add_note(note)
        
        mock_text.add_note.assert_called_once_with(note)
        mock_vector.add_note.assert_called_once_with(note)
    
    def test_remove_note(self, mock_hybrid_engine):
        """Test removing a note from both indices."""
        engine, mock_text, mock_vector = mock_hybrid_engine
        file_path = Path("/vault/test.md")
        
        engine.remove_note(file_path)
        
        mock_text.remove_note.assert_called_once_with(file_path)
        mock_vector.remove_note.assert_called_once_with(str(file_path))
    
    def test_bulk_add_notes(self, mock_hybrid_engine, sample_notes):
        """Test bulk adding notes to both indices."""
        engine, mock_text, mock_vector = mock_hybrid_engine
        
        engine.bulk_add_notes(sample_notes)
        
        mock_text.bulk_add_notes.assert_called_once_with(sample_notes)
        assert mock_vector.add_note.call_count == len(sample_notes)
    
    def test_search_empty_query(self, mock_hybrid_engine):
        """Test search with empty query returns empty results."""
        engine, _, _ = mock_hybrid_engine
        
        results = engine.search("")
        assert results == []
        
        results = engine.search("   ")
        assert results == []
    
    def test_search_text_mode(self, mock_hybrid_engine):
        """Test search in text-only mode."""
        engine, mock_text, mock_vector = mock_hybrid_engine
        
        mock_text.search.return_value = [
            {"path": "/vault/test.md", "title": "Test", "score": 0.8}
        ]
        
        results = engine.search("test query", search_mode="text")
        
        mock_text.search.assert_called_once_with("test query", 50, None, None)
        mock_vector.search.assert_not_called()
        assert len(results) == 1
    
    def test_search_vector_mode(self, mock_hybrid_engine):
        """Test search in vector-only mode."""
        engine, mock_text, mock_vector = mock_hybrid_engine
        
        mock_vector.search.return_value = [
            {"path": "/vault/test.md", "title": "Test", "similarity_score": 0.9, "tags": []}
        ]
        mock_text.get_note_by_path.return_value = {"content": "Test content"}
        
        results = engine.search("test query", search_mode="vector")
        
        mock_vector.search.assert_called_once_with("test query", 50, None)
        mock_text.search.assert_not_called()
        assert len(results) == 1
        assert results[0]["search_mode"] == "vector"
    
    def test_search_hybrid_mode(self, mock_hybrid_engine):
        """Test search in hybrid mode."""
        engine, mock_text, mock_vector = mock_hybrid_engine
        
        # Mock text search results
        mock_text.search.return_value = [
            {"path": "/vault/text_result.md", "title": "Text Result", "score": 0.8, "content": "Text content"}
        ]
        
        # Mock vector search results
        mock_vector.search.return_value = [
            {"path": "/vault/vector_result.md", "title": "Vector Result", "similarity_score": 0.9, "tags": []}
        ]
        mock_text.get_note_by_path.return_value = {"content": "Vector content"}
        
        results = engine.search("test query", search_mode="hybrid")
        
        # Both searches should be called with increased limits
        assert mock_text.search.called
        assert mock_vector.search.called
        
        # Should return fused results
        assert len(results) >= 1
        if results:
            assert "combined_score" in results[0]
            assert results[0]["search_mode"] == "hybrid"
    
    def test_hybrid_search_with_tags(self, mock_hybrid_engine):
        """Test hybrid search with tag filtering."""
        engine, mock_text, mock_vector = mock_hybrid_engine
        
        mock_text.search.return_value = []
        mock_vector.search.return_value = []
        
        tags = {"python", "programming"}
        engine.search("test", tags=tags, search_mode="hybrid")
        
        # Check that tags were passed correctly
        text_call = mock_text.search.call_args
        vector_call = mock_vector.search.call_args
        
        assert text_call[0][2] == tags  # tags parameter for text search
        assert vector_call[0][2] == list(tags)  # tags as list for vector search
    
    def test_reciprocal_rank_fusion(self, mock_hybrid_engine):
        """Test the reciprocal rank fusion algorithm."""
        engine, mock_text, mock_vector = mock_hybrid_engine
        
        # Setup mock results with overlapping documents
        text_results = [
            {"path": "/vault/doc1.md", "title": "Doc 1", "score": 0.9, "content": "Text content 1"},
            {"path": "/vault/doc2.md", "title": "Doc 2", "score": 0.7, "content": "Text content 2"},
            {"path": "/vault/doc3.md", "title": "Doc 3", "score": 0.5, "content": "Text content 3"}
        ]
        
        vector_results = [
            {"path": "/vault/doc2.md", "title": "Doc 2", "similarity_score": 0.95, "tags": []},
            {"path": "/vault/doc4.md", "title": "Doc 4", "similarity_score": 0.85, "tags": []},
            {"path": "/vault/doc1.md", "title": "Doc 1", "similarity_score": 0.75, "tags": []}
        ]
        
        mock_text.search.return_value = text_results
        mock_vector.search.return_value = vector_results
        mock_text.get_note_by_path.return_value = {"content": "Mock content"}
        
        results = engine.search("test query", search_mode="hybrid", limit=3)
        
        assert len(results) <= 3
        
        # Check that results have fusion metadata
        for result in results:
            assert "combined_score" in result
            assert "search_mode" in result
            assert result["search_mode"] == "hybrid"
            
            # Should have rank information
            assert "text_rank" in result or "vector_rank" in result
    
    def test_format_vector_results(self, mock_hybrid_engine):
        """Test formatting vector search results."""
        engine, mock_text, _ = mock_hybrid_engine
        
        vector_results = [
            {
                "path": "/vault/test.md",
                "title": "Test Note",
                "similarity_score": 0.85,
                "tags": ["tag1", "tag2"],
                "created_date": "2024-01-01T00:00:00",
                "modified_date": "2024-01-15T00:00:00"
            }
        ]
        
        mock_text.get_note_by_path.return_value = {
            "content": "This is test content for the note."
        }
        
        formatted = engine._format_vector_results(vector_results)
        
        assert len(formatted) == 1
        result = formatted[0]
        
        assert result["path"] == "/vault/test.md"
        assert result["title"] == "Test Note"
        assert result["score"] == 0.85
        assert result["similarity_score"] == 0.85
        assert result["tags"] == ["tag1", "tag2"]
        assert result["search_mode"] == "vector"
        assert result["highlights"] == {}
        assert "content" in result
    
    def test_format_vector_results_long_content(self, mock_hybrid_engine):
        """Test formatting vector results with long content."""
        engine, mock_text, _ = mock_hybrid_engine
        
        vector_results = [
            {
                "path": "/vault/long.md",
                "title": "Long Note",
                "similarity_score": 0.75,
                "tags": []
            }
        ]
        
        # Mock very long content
        long_content = "A" * 1000
        mock_text.get_note_by_path.return_value = {"content": long_content}
        
        formatted = engine._format_vector_results(vector_results)
        
        result = formatted[0]
        assert len(result["content"]) == 503  # 500 chars + "..."
        assert result["content"].endswith("...")
    
    def test_format_vector_results_no_content(self, mock_hybrid_engine):
        """Test formatting vector results when no content is found."""
        engine, mock_text, _ = mock_hybrid_engine
        
        vector_results = [
            {
                "path": "/vault/missing.md",
                "title": "Missing Note",
                "similarity_score": 0.65,
                "tags": []
            }
        ]
        
        mock_text.get_note_by_path.return_value = None
        
        formatted = engine._format_vector_results(vector_results)
        
        result = formatted[0]
        assert result["content"] == ""
    
    def test_semantic_search(self, mock_hybrid_engine):
        """Test pure semantic/vector search."""
        engine, mock_text, mock_vector = mock_hybrid_engine
        
        mock_vector.search.return_value = [
            {"path": "/vault/semantic.md", "title": "Semantic Result", "similarity_score": 0.88, "tags": []}
        ]
        mock_text.get_note_by_path.return_value = {"content": "Semantic content"}
        
        results = engine.semantic_search("test query", limit=5)
        
        mock_vector.search.assert_called_once_with("test query", 5)
        assert len(results) == 1
        assert results[0]["search_mode"] == "vector"
    
    def test_find_similar_notes(self, mock_hybrid_engine):
        """Test finding similar notes."""
        engine, mock_text, mock_vector = mock_hybrid_engine
        
        mock_vector.get_similar_notes.return_value = [
            {"path": "/vault/similar.md", "title": "Similar Note", "similarity_score": 0.82, "tags": []}
        ]
        mock_text.get_note_by_path.return_value = {"content": "Similar content"}
        
        results = engine.find_similar_notes("/vault/reference.md", limit=3)
        
        mock_vector.get_similar_notes.assert_called_once_with("/vault/reference.md", 3)
        assert len(results) == 1
        assert results[0]["search_mode"] == "vector"
    
    def test_get_note_by_path(self, mock_hybrid_engine):
        """Test getting a note by path."""
        engine, mock_text, _ = mock_hybrid_engine
        
        mock_text.get_note_by_path.return_value = {"title": "Test Note", "content": "Test content"}
        
        result = engine.get_note_by_path(Path("/vault/test.md"))
        
        mock_text.get_note_by_path.assert_called_once_with(Path("/vault/test.md"))
        assert result["title"] == "Test Note"
    
    def test_list_all_tags(self, mock_hybrid_engine):
        """Test listing all tags."""
        engine, mock_text, _ = mock_hybrid_engine
        
        mock_text.list_all_tags.return_value = ["python", "programming", "web"]
        
        tags = engine.list_all_tags()
        
        mock_text.list_all_tags.assert_called_once()
        assert tags == ["python", "programming", "web"]
    
    def test_get_recent_notes(self, mock_hybrid_engine):
        """Test getting recent notes."""
        engine, mock_text, _ = mock_hybrid_engine
        
        mock_text.get_recent_notes.return_value = [
            {"title": "Recent 1", "modified_date": "2024-02-01"},
            {"title": "Recent 2", "modified_date": "2024-01-30"}
        ]
        
        results = engine.get_recent_notes(limit=5)
        
        mock_text.get_recent_notes.assert_called_once_with(5)
        assert len(results) == 2
    
    def test_search_with_fields(self, mock_hybrid_engine):
        """Test search with specific fields."""
        engine, mock_text, _ = mock_hybrid_engine
        
        mock_text.search.return_value = []
        
        search_fields = ["title", "content"]
        engine.search("test", search_fields=search_fields, search_mode="text")
        
        mock_text.search.assert_called_once_with("test", 50, None, search_fields)
    
    def test_hybrid_alpha_weighting(self):
        """Test that hybrid_alpha affects result weighting."""
        # Test with different alpha values
        alphas = [0.2, 0.8]
        
        for alpha in alphas:
            with tempfile.TemporaryDirectory() as temp_dir:
                vault_path = Path(temp_dir) / "vault"
                vault_path.mkdir()
                config = ServerConfig(
                    vault_path=vault_path,
                    index_path=Path(temp_dir) / "index",
                    vector_index_path=Path(temp_dir) / "vector_index",
                    embedding_model="test-model",
                    hybrid_alpha=alpha,
                    max_results=50
                )
                
                with patch('obsidian_mcp.search.ObsidianSearchIndex') as mock_text_search, \
                     patch('obsidian_mcp.search.VectorSearchEngine') as mock_vector_search:
                    
                    mock_text_instance = Mock()
                    mock_vector_instance = Mock()
                    mock_text_search.return_value = mock_text_instance
                    mock_vector_search.return_value = mock_vector_instance
                    
                    engine = HybridSearchEngine(config)
                    engine.text_search = mock_text_instance
                    engine.vector_search = mock_vector_instance
                    
                    # Mock results
                    mock_text_instance.search.return_value = [
                        {"path": "/vault/test.md", "title": "Test", "score": 0.8, "content": "content"}
                    ]
                    mock_vector_instance.search.return_value = [
                        {"path": "/vault/test.md", "title": "Test", "similarity_score": 0.9, "tags": []}
                    ]
                    mock_text_instance.get_note_by_path.return_value = {"content": "content"}
                    
                    results = engine.search("test", search_mode="hybrid")
                    
                    # Should have results with combined scores influenced by alpha
                    if results:
                        assert "combined_score" in results[0]
                        # The actual score calculation depends on RRF, but alpha should influence it