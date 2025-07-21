"""Tests for the Obsidian parser module."""

import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from obsidian_mcp.parser import ObsidianParser, ObsidianNote


@pytest.fixture
def temp_vault():
    """Create a temporary vault for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        
        # Create test notes
        (vault_path / "note1.md").write_text("""---
title: Test Note 1
tags: [python, testing]
---

# Test Note 1

This is a test note with some content.

See also [[note2]] and [[another note]].

#test-tag #python
""")
        
        (vault_path / "note2.md").write_text("""# Note 2

This is another note without frontmatter.

It links to [[note1]].
""")
        
        (vault_path / "subfolder").mkdir()
        (vault_path / "subfolder" / "nested.md").write_text("""---
tags: nested, folder
---

# Nested Note

This note is in a subfolder.
""")
        
        yield vault_path


def test_parser_initialization(temp_vault):
    """Test parser initialization."""
    parser = ObsidianParser(temp_vault)
    assert parser.vault_path == temp_vault
    assert isinstance(parser._note_cache, dict)


def test_discover_notes(temp_vault):
    """Test note discovery."""
    parser = ObsidianParser(temp_vault)
    notes = parser.discover_notes()
    
    assert len(notes) == 3
    note_names = [note.name for note in notes]
    assert "note1.md" in note_names
    assert "note2.md" in note_names
    assert "nested.md" in note_names


def test_parse_note_with_frontmatter(temp_vault):
    """Test parsing a note with frontmatter."""
    parser = ObsidianParser(temp_vault)
    note_path = temp_vault / "note1.md"
    
    note = parser.parse_note(note_path)
    
    assert note is not None
    assert note.title == "Test Note 1"
    assert note.path == note_path
    assert "python" in note.tags
    assert "testing" in note.tags
    assert "test-tag" in note.tags
    assert "note2" in note.wikilinks
    assert "another note" in note.wikilinks
    assert "This is a test note" in note.content


def test_parse_note_without_frontmatter(temp_vault):
    """Test parsing a note without frontmatter."""
    parser = ObsidianParser(temp_vault)
    note_path = temp_vault / "note2.md"
    
    note = parser.parse_note(note_path)
    
    assert note is not None
    assert note.title == "Note 2"  # From first heading
    assert note.path == note_path
    assert len(note.tags) == 0
    assert "note1" in note.wikilinks
    assert "This is another note" in note.content


def test_extract_title_from_filename(temp_vault):
    """Test title extraction from filename when no heading exists."""
    parser = ObsidianParser(temp_vault)
    
    # Create a note without headings
    note_path = temp_vault / "no-heading.md"
    note_path.write_text("Just some content without a heading.")
    
    note = parser.parse_note(note_path)
    assert note.title == "no-heading"


def test_tag_extraction_formats(temp_vault):
    """Test various tag extraction formats."""
    parser = ObsidianParser(temp_vault)
    
    # Test comma-separated tags in frontmatter
    note_path = temp_vault / "tag-test.md"
    note_path.write_text("""---
tags: tag1, tag2, tag3
---

Content with #inline-tag and #another/nested/tag.
""")
    
    note = parser.parse_note(note_path)
    expected_tags = {"tag1", "tag2", "tag3", "inline-tag", "another/nested/tag"}
    assert note.tags == expected_tags


def test_compute_backlinks(temp_vault):
    """Test backlink computation."""
    parser = ObsidianParser(temp_vault)
    
    # Parse all notes
    note_paths = parser.discover_notes()
    notes = []
    for path in note_paths:
        note = parser.parse_note(path)
        if note:
            notes.append(note)
    
    # Compute backlinks
    parser.compute_backlinks(notes)
    
    # Find note1 and check its backlinks
    note1 = next(note for note in notes if note.path.name == "note1.md")
    assert "note2" in note1.backlinks


def test_get_note_by_name(temp_vault):
    """Test getting notes by name."""
    parser = ObsidianParser(temp_vault)
    
    # Parse a note first
    note_path = temp_vault / "note1.md"
    note = parser.parse_note(note_path)
    
    # Test retrieval by filename
    retrieved = parser.get_note_by_name("note1")
    assert retrieved is not None
    assert retrieved.path == note_path
    
    # Test retrieval by title
    retrieved = parser.get_note_by_name("Test Note 1")
    assert retrieved is not None
    assert retrieved.path == note_path


def test_wikilink_extraction_with_display_text(temp_vault):
    """Test wikilink extraction with display text."""
    parser = ObsidianParser(temp_vault)
    
    note_path = temp_vault / "wikilink-test.md"
    note_path.write_text("""
# Wikilink Test

This links to [[target note|Display Text]] and [[another note]].
""")
    
    note = parser.parse_note(note_path)
    assert "target note" in note.wikilinks
    assert "another note" in note.wikilinks


def test_ignore_obsidian_folder(temp_vault):
    """Test that .obsidian folder is ignored."""
    parser = ObsidianParser(temp_vault)
    
    # Create .obsidian folder with files
    obsidian_dir = temp_vault / ".obsidian"
    obsidian_dir.mkdir()
    (obsidian_dir / "config.md").write_text("Config file")
    
    notes = parser.discover_notes()
    note_paths = [str(note) for note in notes]
    
    # Ensure no .obsidian files are included
    assert not any(".obsidian" in path for path in note_paths)