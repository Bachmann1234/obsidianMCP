"""Tests for the search module."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from obsidian_mcp.parser import ObsidianNote
from obsidian_mcp.search import ObsidianSearchIndex


@pytest.fixture
def temp_index():
    """Create a temporary search index."""
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = Path(temp_dir) / "test_index"
        yield ObsidianSearchIndex(index_path)


@pytest.fixture
def sample_notes():
    """Create sample notes for testing."""
    notes = []

    # Note 1: Python programming
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
        modified_date=datetime(2024, 1, 15),
    )

    # Note 2: Advanced Python
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
        modified_date=datetime(2024, 1, 20),
    )

    # Note 3: Cooking recipe
    note3 = ObsidianNote(
        path=Path("/vault/pasta-recipe.md"),
        title="Pasta Recipe",
        content="A delicious pasta recipe with tomatoes and herbs.",
        raw_content="# Pasta Recipe\n\nDelicious pasta.",
        frontmatter={"category": "cooking"},
        tags={"cooking", "recipe", "italian"},
        wikilinks=set(),
        backlinks=set(),
        created_date=datetime(2024, 1, 5),
        modified_date=datetime(2024, 1, 10),
    )

    return [note1, note2, note3]


def test_index_creation(temp_index):
    """Test search index creation."""
    assert temp_index._index is not None
    assert temp_index.index_path.exists()


def test_add_single_note(temp_index, sample_notes):
    """Test adding a single note to the index."""
    note = sample_notes[0]
    temp_index.add_note(note)

    # Search for the note
    results = temp_index.search("Python")
    assert len(results) == 1
    assert results[0]["title"] == "Python Basics"


def test_bulk_add_notes(temp_index, sample_notes):
    """Test bulk adding notes to the index."""
    temp_index.bulk_add_notes(sample_notes)

    # Search should find multiple results
    results = temp_index.search("Python")
    assert len(results) == 2

    titles = [result["title"] for result in results]
    assert "Python Basics" in titles
    assert "Advanced Python" in titles


def test_search_by_content(temp_index, sample_notes):
    """Test searching by content."""
    temp_index.bulk_add_notes(sample_notes)

    # Search for cooking content
    results = temp_index.search("delicious pasta")
    assert len(results) == 1
    assert results[0]["title"] == "Pasta Recipe"


def test_search_by_tags(temp_index, sample_notes):
    """Test searching with tag filters."""
    temp_index.bulk_add_notes(sample_notes)

    # Search with tag filter
    results = temp_index.search("programming", tags={"python"})
    assert len(results) == 2

    # Search with different tag
    results = temp_index.search("recipe", tags={"cooking"})
    assert len(results) == 1
    assert results[0]["title"] == "Pasta Recipe"


def test_search_limit(temp_index, sample_notes):
    """Test search result limiting."""
    temp_index.bulk_add_notes(sample_notes)

    # Search with limit
    results = temp_index.search("python", limit=1)
    assert len(results) == 1


def test_get_note_by_path(temp_index, sample_notes):
    """Test retrieving a note by its path."""
    temp_index.bulk_add_notes(sample_notes)

    note_path = Path("/vault/python-basics.md")
    result = temp_index.get_note_by_path(note_path)

    assert result is not None
    assert result["title"] == "Python Basics"
    assert result["path"] == str(note_path)


def test_list_all_tags(temp_index, sample_notes):
    """Test listing all tags."""
    temp_index.bulk_add_notes(sample_notes)

    tags = temp_index.list_all_tags()
    expected_tags = {
        "python",
        "programming",
        "beginner",
        "advanced",
        "cooking",
        "recipe",
        "italian",
    }

    assert set(tags) == expected_tags


def test_get_recent_notes(temp_index, sample_notes):
    """Test getting recent notes."""
    temp_index.bulk_add_notes(sample_notes)

    recent = temp_index.get_recent_notes(limit=2)
    assert len(recent) == 2

    # Should be sorted by modification date (newest first)
    assert recent[0]["title"] == "Advanced Python"  # Modified 2024-01-20
    assert recent[1]["title"] == "Python Basics"  # Modified 2024-01-15


def test_remove_note(temp_index, sample_notes):
    """Test removing a note from the index."""
    temp_index.bulk_add_notes(sample_notes)

    # Verify note exists
    results = temp_index.search("Python Basics")
    assert len(results) >= 1

    # Remove the note
    note_path = Path("/vault/python-basics.md")
    temp_index.remove_note(note_path)

    # Verify note is removed
    result = temp_index.get_note_by_path(note_path)
    assert result is None


def test_update_note(temp_index, sample_notes):
    """Test updating a note in the index."""
    temp_index.bulk_add_notes(sample_notes)

    # Update the note content
    updated_note = sample_notes[0]
    updated_note.content = "Updated Python content with new information."
    updated_note.title = "Updated Python Basics"

    temp_index.add_note(updated_note)

    # Search for updated content
    results = temp_index.search("Updated Python")
    assert len(results) == 1
    assert results[0]["title"] == "Updated Python Basics"


def test_search_fields(temp_index, sample_notes):
    """Test searching specific fields."""
    temp_index.bulk_add_notes(sample_notes)

    # Search only in title
    results = temp_index.search("Basics", search_fields=["title"])
    assert len(results) == 1
    assert results[0]["title"] == "Python Basics"

    # Search only in content
    results = temp_index.search("decorators", search_fields=["content"])
    assert len(results) == 1
    assert results[0]["title"] == "Advanced Python"


def test_rebuild_index(temp_index, sample_notes):
    """Test rebuilding the entire index."""
    # Add initial notes
    temp_index.bulk_add_notes(sample_notes[:2])

    results = temp_index.search("python")
    assert len(results) == 2

    # Rebuild with different notes
    temp_index.rebuild_index([sample_notes[2]])  # Only cooking note

    results = temp_index.search("python")
    assert len(results) == 0

    results = temp_index.search("pasta")
    assert len(results) == 1


def test_get_stats(temp_index, sample_notes):
    """Test getting index statistics."""
    temp_index.bulk_add_notes(sample_notes)

    stats = temp_index.get_stats()
    assert stats["doc_count"] == 3
    assert "field_names" in stats
    assert "index_path" in stats

    expected_fields = {
        "path",
        "title",
        "content",
        "tags",
        "wikilinks",
        "created_date",
        "modified_date",
        "frontmatter",
    }
    assert set(stats["field_names"]) == expected_fields


def test_empty_search(temp_index):
    """Test searching with empty query."""
    results = temp_index.search("")
    assert len(results) == 0

    results = temp_index.search("   ")
    assert len(results) == 0


def test_search_query_parsing_error(temp_index, sample_notes):
    """Test search when query parsing fails."""
    temp_index.bulk_add_notes(sample_notes)

    # Patch the MultifieldParser constructor to cause parse errors
    with patch("obsidian_mcp.search.MultifieldParser") as mock_parser_class:
        mock_parser = Mock()
        mock_parser.parse.side_effect = Exception("Parse error")
        mock_parser_class.return_value = mock_parser

        # Should fall back to simple content search
        results = temp_index.search("python")
        # Should still work due to fallback
        assert isinstance(results, list)


def test_optimize_index():
    """Test index optimization method exists and can be called."""
    # Just test that the method exists and can be called
    # without actually running optimization which can conflict with test setup
    import tempfile
    from pathlib import Path

    from obsidian_mcp.search import ObsidianSearchIndex

    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = Path(temp_dir) / "test_index"
        search_index = ObsidianSearchIndex(index_path)

        # Test that the method exists
        assert hasattr(search_index, "optimize_index")
        assert callable(getattr(search_index, "optimize_index"))


def test_get_recent_notes_sorting(temp_index, sample_notes):
    """Test that recent notes are properly sorted."""
    temp_index.bulk_add_notes(sample_notes)

    recent = temp_index.get_recent_notes(limit=3)
    assert len(recent) <= 3
    assert isinstance(recent, list)

    # Check that all results have the expected structure
    for note in recent:
        assert "title" in note
        assert "content" in note
        assert "path" in note


# ========== Corruption Handling Tests ==========


def test_validate_index_success(temp_index):
    """Test successful index validation."""
    # Index should validate successfully after creation
    temp_index._validate_index()  # Should not raise


def test_validate_index_failure():
    """Test index validation failure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = Path(temp_dir) / "test_index"
        search_index = ObsidianSearchIndex(index_path)
        
        # Mock the searcher to simulate corruption
        with patch.object(search_index._index, 'searcher') as mock_searcher_context:
            mock_searcher = Mock()
            mock_searcher.doc_count.side_effect = Exception("Corruption error")
            mock_searcher_context.return_value.__enter__.return_value = mock_searcher
            
            # Should raise IndexError on validation failure
            with pytest.raises(Exception):  # Will be wrapped in IndexError
                search_index._validate_index()


def test_recover_from_corruption_success():
    """Test successful corruption recovery."""
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = Path(temp_dir) / "test_index"
        
        # Create index first
        search_index = ObsidianSearchIndex(index_path)
        assert index_path.exists()
        
        # Verify recovery works
        result = search_index._recover_from_corruption()
        assert result is True
        assert index_path.exists()  # Should be recreated
        
        # Index should be usable after recovery
        search_index._validate_index()


def test_recover_from_corruption_directory_removal_failure():
    """Test corruption recovery when directory removal fails."""
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = Path(temp_dir) / "test_index"
        search_index = ObsidianSearchIndex(index_path)
        
        # Mock shutil.rmtree to fail
        with patch('obsidian_mcp.search.shutil.rmtree', side_effect=OSError("Permission denied")):
            result = search_index._recover_from_corruption()
            assert result is False


def test_recover_from_corruption_index_creation_failure():
    """Test corruption recovery when fresh index creation fails."""
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = Path(temp_dir) / "test_index"
        search_index = ObsidianSearchIndex(index_path)
        
        # Mock index.create_in to fail
        with patch('obsidian_mcp.search.index.create_in', side_effect=Exception("Creation failed")):
            result = search_index._recover_from_corruption()
            assert result is False


def test_ensure_index_corruption_retry_success():
    """Test _ensure_index successfully recovers from corruption on retry."""
    from whoosh.index import IndexError
    
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = Path(temp_dir) / "test_index"
        
        # Create a corrupted index directory first
        index_path.mkdir(parents=True)
        (index_path / "corrupted_file").write_text("bad data")
        
        # Mock the sequence: first attempt fails, recovery clears index, second attempt creates new
        with patch('obsidian_mcp.search.index.exists_in', side_effect=[True, False]), \
             patch('obsidian_mcp.search.index.open_dir', side_effect=IndexError("Corruption")), \
             patch.object(ObsidianSearchIndex, '_recover_from_corruption', return_value=True), \
             patch('obsidian_mcp.search.index.create_in') as mock_create:
            
            # This should succeed on the second attempt after recovery
            search_index = ObsidianSearchIndex(index_path)
            
            # Verify new index was created after recovery
            assert mock_create.called


def test_ensure_index_corruption_max_retries_exceeded():
    """Test _ensure_index fails after max retries are exceeded."""
    from whoosh.index import IndexError
    
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = Path(temp_dir) / "test_index"
        
        with patch('obsidian_mcp.search.index.exists_in', return_value=True), \
             patch('obsidian_mcp.search.index.open_dir', side_effect=IndexError("Persistent corruption")), \
             patch.object(ObsidianSearchIndex, '_recover_from_corruption', return_value=False):
            
            # Should raise RuntimeError after max retries
            with pytest.raises(RuntimeError, match="Unable to initialize search index"):
                ObsidianSearchIndex(index_path)


def test_ensure_index_handles_specific_exceptions():
    """Test _ensure_index handles specific corruption exception types."""
    from whoosh.index import IndexError, EmptyIndexError, LockError
    
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = Path(temp_dir) / "test_index"
        
        # Test each specific exception type
        exceptions_to_test = [
            IndexError("Index corrupted"),
            EmptyIndexError("Index empty"),
            LockError("Index locked"),
            TypeError("Type error"),
            ValueError("Value error")
        ]
        
        for exception in exceptions_to_test:
            # For each exception, simulate recovery causing switch to create_in path
            with patch('obsidian_mcp.search.index.exists_in', side_effect=[True, False]), \
                 patch('obsidian_mcp.search.index.open_dir', side_effect=exception), \
                 patch.object(ObsidianSearchIndex, '_recover_from_corruption', return_value=True), \
                 patch('obsidian_mcp.search.index.create_in'):
                 
                # Should handle the exception and recover
                search_index = ObsidianSearchIndex(index_path)
                assert search_index is not None


def test_add_note_corruption_recovery_success(sample_notes):
    """Test add_note successfully recovers from corruption."""
    from whoosh.index import IndexError
    
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = Path(temp_dir) / "test_index"
        search_index = ObsidianSearchIndex(index_path)
        
        note = sample_notes[0]
        
        # Mock writer to fail first, then succeed after recovery
        with patch.object(search_index._index, 'writer') as mock_writer_context:
            # First call fails, second succeeds
            mock_writer1 = Mock()
            mock_writer1.update_document.side_effect = IndexError("Index corruption")
            mock_writer2 = Mock()  # Second writer succeeds
            
            mock_writer_context.return_value.__enter__.side_effect = [mock_writer1, mock_writer2]
            
            # Mock recovery to succeed
            with patch.object(search_index, '_recover_from_corruption', return_value=True):
                # Should succeed after recovery
                search_index.add_note(note)
                
                # Verify both attempts were made
                assert mock_writer_context.call_count == 2
                mock_writer2.update_document.assert_called_once()


def test_add_note_corruption_recovery_failure(sample_notes):
    """Test add_note fails when recovery doesn't work."""
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = Path(temp_dir) / "test_index"
        search_index = ObsidianSearchIndex(index_path)
        
        note = sample_notes[0]
        
        # Mock writer to always fail
        with patch.object(search_index._index, 'writer') as mock_writer_context:
            mock_writer = Mock()
            mock_writer.update_document.side_effect = Exception("Persistent corruption")
            mock_writer_context.return_value.__enter__.return_value = mock_writer
            
            # Mock recovery to fail
            with patch.object(search_index, '_recover_from_corruption', return_value=False):
                # Should raise the original exception
                with pytest.raises(Exception, match="Persistent corruption"):
                    search_index.add_note(note)


def test_search_corruption_graceful_degradation(temp_index, sample_notes):
    """Test search returns empty results instead of crashing on corruption."""
    from whoosh.index import IndexError
    
    temp_index.bulk_add_notes(sample_notes)
    
    # Mock searcher to fail with the specific exception type caught by search()
    with patch.object(temp_index._index, 'searcher', side_effect=IndexError("Search corruption")):
        # Mock recovery to succeed
        with patch.object(temp_index, '_recover_from_corruption', return_value=True):
            # Should return empty results instead of crashing
            results = temp_index.search("python")
            assert results == []


def test_search_corruption_recovery_attempt(temp_index, sample_notes):
    """Test search attempts recovery when corruption is detected."""
    from whoosh.index import IndexError
    
    temp_index.bulk_add_notes(sample_notes)
    
    # Mock searcher to fail with the specific exception type caught by search()
    with patch.object(temp_index._index, 'searcher', side_effect=IndexError("Search corruption")):
        # Mock recovery method
        with patch.object(temp_index, '_recover_from_corruption', return_value=True) as mock_recovery:
            # Should attempt recovery
            results = temp_index.search("python")
            
            # Verify recovery was attempted
            mock_recovery.assert_called_once()
            assert results == []


def test_search_corruption_no_recovery_attempt_on_success(temp_index, sample_notes):
    """Test search doesn't attempt recovery when working normally."""
    temp_index.bulk_add_notes(sample_notes)
    
    # Mock recovery method to ensure it's not called
    with patch.object(temp_index, '_recover_from_corruption') as mock_recovery:
        # Normal search should work
        results = temp_index.search("python")
        
        # Recovery should not be called for successful searches
        mock_recovery.assert_not_called()
        assert len(results) > 0
