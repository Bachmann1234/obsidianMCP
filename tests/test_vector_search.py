"""Tests for the vector search module."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from obsidian_mcp.parser import ObsidianNote
from obsidian_mcp.vector_search import VectorSearchEngine


@pytest.fixture
def temp_vector_index():
    """Create a temporary vector search index."""
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = Path(temp_dir) / "test_vector_index"
        # Mock ChromaDB and SentenceTransformers to avoid downloading models
        with (
            patch(
                "obsidian_mcp.vector_search.chromadb.PersistentClient"
            ) as mock_client,
            patch("obsidian_mcp.vector_search.SentenceTransformer") as mock_transformer,
        ):
            # Mock ChromaDB client and collection
            mock_collection = Mock()
            mock_client_instance = Mock()
            mock_client_instance.get_or_create_collection.return_value = mock_collection
            mock_client_instance.delete_collection.return_value = None
            mock_client_instance.create_collection.return_value = mock_collection
            mock_client.return_value = mock_client_instance

            # Mock SentenceTransformer
            mock_transformer_instance = Mock()
            mock_transformer_instance.encode.return_value = np.array(
                [0.1, 0.2, 0.3, 0.4, 0.5]
            )
            mock_transformer.return_value = mock_transformer_instance

            engine = VectorSearchEngine(index_path)
            engine.collection = mock_collection
            engine._embedding_model = mock_transformer_instance

            yield engine, mock_collection, mock_transformer_instance


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
        modified_date=datetime(2024, 2, 1),
    )

    # Note 3: JavaScript
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
        modified_date=datetime(2024, 2, 10),
    )

    return [note1, note2, note3]


class TestVectorSearchEngine:
    """Test the VectorSearchEngine class."""

    def test_init(self, temp_vector_index):
        """Test vector search engine initialization."""
        engine, mock_collection, mock_transformer = temp_vector_index

        assert engine.embedding_model_name == "all-MiniLM-L6-v2"
        assert engine.collection_name == "obsidian_notes"
        assert engine.index_path.name == "test_vector_index"

    def test_embedding_model_lazy_loading(self):
        """Test that embedding model is loaded lazily."""
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "test_lazy"

            with (
                patch("obsidian_mcp.vector_search.chromadb.PersistentClient"),
                patch(
                    "obsidian_mcp.vector_search.SentenceTransformer"
                ) as mock_transformer,
            ):
                mock_transformer_instance = Mock()
                mock_transformer.return_value = mock_transformer_instance

                engine = VectorSearchEngine(index_path)

                # Model should not be loaded yet
                assert engine._embedding_model is None

                # Access the property to trigger lazy loading
                model = engine.embedding_model

                # Model should now be loaded
                assert engine._embedding_model is mock_transformer_instance
                assert model is mock_transformer_instance
                mock_transformer.assert_called_once_with("all-MiniLM-L6-v2")

    def test_generate_doc_id(self, temp_vector_index):
        """Test document ID generation."""
        engine, _, _ = temp_vector_index

        doc_id1 = engine._generate_doc_id("/vault/test.md")
        doc_id2 = engine._generate_doc_id("/vault/test.md")
        doc_id3 = engine._generate_doc_id("/vault/other.md")

        # Same path should generate same ID
        assert doc_id1 == doc_id2
        # Different paths should generate different IDs
        assert doc_id1 != doc_id3
        # IDs should be 32-character hex strings (MD5)
        assert len(doc_id1) == 32
        assert all(c in "0123456789abcdef" for c in doc_id1)

    def test_prepare_text_for_embedding(self, temp_vector_index, sample_notes):
        """Test text preparation for embedding."""
        engine, _, _ = temp_vector_index
        note = sample_notes[0]

        text = engine._prepare_text_for_embedding(note)

        assert "Title: Python Basics" in text
        assert "Content: Python is a programming language" in text
        assert "Tags: " in text
        assert "python" in text
        assert "programming" in text
        assert "beginner" in text

    def test_prepare_text_for_embedding_minimal_note(self, temp_vector_index):
        """Test text preparation with minimal note data."""
        engine, _, _ = temp_vector_index

        note = ObsidianNote(
            path=Path("/vault/minimal.md"),
            title="",
            content="",
            raw_content="",
            frontmatter={},
            tags=set(),
            wikilinks=set(),
            backlinks=set(),
            created_date=None,
            modified_date=None,
        )

        text = engine._prepare_text_for_embedding(note)
        assert text == ""

    def test_add_note_new(self, temp_vector_index, sample_notes):
        """Test adding a new note to the vector index."""
        engine, mock_collection, _ = temp_vector_index
        note = sample_notes[0]

        # Mock that note doesn't exist
        mock_collection.get.return_value = {"ids": []}

        engine.add_note(note)

        # Verify that get was called to check existence
        mock_collection.get.assert_called_once()

        # Verify that add was called
        mock_collection.add.assert_called_once()
        add_call = mock_collection.add.call_args

        # Check the call arguments
        assert len(add_call[1]["ids"]) == 1
        assert len(add_call[1]["embeddings"]) == 1
        assert len(add_call[1]["metadatas"]) == 1
        assert len(add_call[1]["documents"]) == 1

        metadata = add_call[1]["metadatas"][0]
        assert metadata["path"] == str(note.path)
        assert metadata["title"] == note.title
        assert json.loads(metadata["tags"]) == list(note.tags)

    def test_add_note_update_existing(self, temp_vector_index, sample_notes):
        """Test updating an existing note in the vector index."""
        engine, mock_collection, _ = temp_vector_index
        note = sample_notes[0]

        # Mock that note exists
        mock_collection.get.return_value = {"ids": ["existing_id"]}

        engine.add_note(note)

        # Verify that update was called instead of add
        mock_collection.update.assert_called_once()
        mock_collection.add.assert_not_called()

    def test_add_note_error_handling(self, temp_vector_index, sample_notes):
        """Test error handling when adding a note fails."""
        engine, mock_collection, _ = temp_vector_index
        note = sample_notes[0]

        # Mock an error during embedding generation
        engine._embedding_model.encode.side_effect = Exception("Encoding failed")

        # Should not raise exception
        engine.add_note(note)

        # Collection methods should not have been called
        mock_collection.get.assert_not_called()
        mock_collection.add.assert_not_called()

    def test_remove_note_existing(self, temp_vector_index):
        """Test removing an existing note from the vector index."""
        engine, mock_collection, _ = temp_vector_index
        note_path = "/vault/test.md"

        # Mock that note exists
        mock_collection.get.return_value = {"ids": ["existing_id"]}

        engine.remove_note(note_path)

        # Verify delete was called
        mock_collection.delete.assert_called_once()

    def test_remove_note_non_existing(self, temp_vector_index):
        """Test removing a non-existing note from the vector index."""
        engine, mock_collection, _ = temp_vector_index
        note_path = "/vault/nonexistent.md"

        # Mock that note doesn't exist
        mock_collection.get.return_value = {"ids": []}

        engine.remove_note(note_path)

        # Verify delete was not called
        mock_collection.delete.assert_not_called()

    def test_remove_note_error_handling(self, temp_vector_index):
        """Test error handling when removing a note fails."""
        engine, mock_collection, _ = temp_vector_index
        note_path = "/vault/test.md"

        # Mock an error during get
        mock_collection.get.side_effect = Exception("Get failed")

        # Should not raise exception
        engine.remove_note(note_path)

        # Delete should not have been called
        mock_collection.delete.assert_not_called()

    def test_search_basic(self, temp_vector_index):
        """Test basic vector search functionality."""
        engine, mock_collection, _ = temp_vector_index

        # Mock search results
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "metadatas": [
                [
                    {
                        "path": "/vault/note1.md",
                        "title": "Note 1",
                        "tags": '["python", "programming"]',
                        "created_date": "2024-01-01T00:00:00",
                        "modified_date": "2024-01-15T00:00:00",
                        "content_length": 100,
                    },
                    {
                        "path": "/vault/note2.md",
                        "title": "Note 2",
                        "tags": '["javascript", "web"]',
                        "created_date": "2024-02-01T00:00:00",
                        "modified_date": "2024-02-10T00:00:00",
                        "content_length": 150,
                    },
                ]
            ],
            "distances": [[0.2, 0.4]],
        }

        results = engine.search("python programming", top_k=5)

        assert len(results) == 2
        assert results[0]["path"] == "/vault/note1.md"
        assert results[0]["title"] == "Note 1"
        assert results[0]["similarity_score"] == 0.8  # 1.0 - 0.2
        assert results[1]["similarity_score"] == 0.6  # 1.0 - 0.4

        # Verify query was called correctly
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args[1]
        assert len(call_args["query_embeddings"]) == 1
        assert call_args["n_results"] == 10  # top_k * 2

    def test_search_with_tag_filter(self, temp_vector_index):
        """Test vector search with tag filtering."""
        engine, mock_collection, _ = temp_vector_index

        # Mock search results with different tags
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2", "doc3"]],
            "metadatas": [
                [
                    {
                        "path": "/vault/python_note.md",
                        "title": "Python Note",
                        "tags": '["python", "programming"]',
                        "created_date": "2024-01-01T00:00:00",
                        "modified_date": "2024-01-15T00:00:00",
                        "content_length": 100,
                    },
                    {
                        "path": "/vault/js_note.md",
                        "title": "JS Note",
                        "tags": '["javascript", "web"]',
                        "created_date": "2024-02-01T00:00:00",
                        "modified_date": "2024-02-10T00:00:00",
                        "content_length": 150,
                    },
                    {
                        "path": "/vault/python_advanced.md",
                        "title": "Advanced Python",
                        "tags": '["python", "advanced"]',
                        "created_date": "2024-03-01T00:00:00",
                        "modified_date": "2024-03-10T00:00:00",
                        "content_length": 200,
                    },
                ]
            ],
            "distances": [[0.1, 0.2, 0.3]],
        }

        # Search with python tag filter
        results = engine.search("programming", top_k=5, tag_filter=["python"])

        # Should only return notes with python tag
        assert len(results) == 2
        assert results[0]["path"] == "/vault/python_note.md"
        assert results[1]["path"] == "/vault/python_advanced.md"
        # JS note should be filtered out
        assert not any(r["path"] == "/vault/js_note.md" for r in results)

    def test_search_error_handling(self, temp_vector_index):
        """Test error handling during search."""
        engine, mock_collection, _ = temp_vector_index

        # Mock an error during query
        mock_collection.query.side_effect = Exception("Query failed")

        results = engine.search("test query")

        # Should return empty list instead of raising exception
        assert results == []

    def test_get_similar_notes_existing(self, temp_vector_index):
        """Test finding similar notes for an existing note."""
        engine, mock_collection, _ = temp_vector_index
        note_path = "/vault/reference.md"

        # Generate the correct doc ID for the reference note
        ref_doc_id = engine._generate_doc_id(note_path)

        # Mock getting the reference note's embedding
        mock_collection.get.return_value = {
            "ids": [ref_doc_id],
            "embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5]],
        }

        # Mock search results with the correct ref doc ID
        mock_collection.query.return_value = {
            "ids": [[ref_doc_id, "similar1", "similar2"]],
            "metadatas": [
                [
                    {
                        "path": "/vault/reference.md",
                        "title": "Reference",
                        "tags": "[]",
                        "created_date": "",
                        "modified_date": "",
                    },
                    {
                        "path": "/vault/similar1.md",
                        "title": "Similar 1",
                        "tags": '["tag1"]',
                        "created_date": "",
                        "modified_date": "",
                    },
                    {
                        "path": "/vault/similar2.md",
                        "title": "Similar 2",
                        "tags": '["tag2"]',
                        "created_date": "",
                        "modified_date": "",
                    },
                ]
            ],
            "distances": [[0.0, 0.3, 0.5]],
        }

        results = engine.get_similar_notes(note_path, top_k=5)

        # Should exclude the reference note itself
        assert len(results) == 2
        assert not any(r["path"] == "/vault/reference.md" for r in results)
        assert results[0]["path"] == "/vault/similar1.md"
        assert results[1]["path"] == "/vault/similar2.md"

    def test_get_similar_notes_not_found(self, temp_vector_index):
        """Test finding similar notes when reference note doesn't exist."""
        engine, mock_collection, _ = temp_vector_index
        note_path = "/vault/nonexistent.md"

        # Mock that note doesn't exist
        mock_collection.get.return_value = {"ids": []}

        results = engine.get_similar_notes(note_path)

        assert results == []
        # Query should not have been called
        mock_collection.query.assert_not_called()

    def test_get_similar_notes_error_handling(self, temp_vector_index):
        """Test error handling when finding similar notes fails."""
        engine, mock_collection, _ = temp_vector_index
        note_path = "/vault/test.md"

        # Mock an error during get
        mock_collection.get.side_effect = Exception("Get failed")

        results = engine.get_similar_notes(note_path)

        assert results == []

    def test_get_stats_success(self, temp_vector_index):
        """Test getting vector index statistics."""
        engine, mock_collection, _ = temp_vector_index

        # Mock collection count
        mock_collection.count.return_value = 42

        stats = engine.get_stats()

        assert stats["total_notes"] == 42
        assert stats["embedding_model"] == "all-MiniLM-L6-v2"
        assert stats["collection_name"] == "obsidian_notes"
        assert "index_path" in stats

    def test_get_stats_error(self, temp_vector_index):
        """Test getting statistics when an error occurs."""
        engine, mock_collection, _ = temp_vector_index

        # Mock an error during count
        mock_collection.count.side_effect = Exception("Count failed")

        stats = engine.get_stats()

        assert stats["total_notes"] == 0
        assert stats["embedding_model"] == "all-MiniLM-L6-v2"
        assert "error" in stats

    def test_rebuild_index(self, temp_vector_index, sample_notes):
        """Test rebuilding the entire vector index."""
        engine, mock_collection, _ = temp_vector_index

        # Mock client methods
        engine.client.delete_collection = Mock()
        engine.client.create_collection = Mock(return_value=mock_collection)

        # Mock collection methods for adding notes
        mock_collection.get.return_value = {"ids": []}  # All notes are new

        engine.rebuild_index(sample_notes)

        # Verify collection was deleted and recreated
        engine.client.delete_collection.assert_called_once_with("obsidian_notes")
        engine.client.create_collection.assert_called_once()

        # Verify all notes were added
        assert mock_collection.add.call_count == len(sample_notes)

    def test_rebuild_index_error_handling(self, temp_vector_index, sample_notes):
        """Test error handling during index rebuild."""
        engine, mock_collection, _ = temp_vector_index

        # Mock an error during delete_collection
        engine.client.delete_collection.side_effect = Exception("Delete failed")

        # Should raise the exception
        with pytest.raises(Exception, match="Delete failed"):
            engine.rebuild_index(sample_notes)

    def test_custom_embedding_model(self):
        """Test initializing with a custom embedding model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "custom_model_test"

            with (
                patch("obsidian_mcp.vector_search.chromadb.PersistentClient"),
                patch(
                    "obsidian_mcp.vector_search.SentenceTransformer"
                ) as mock_transformer,
            ):
                custom_model = "custom-model-name"
                engine = VectorSearchEngine(index_path, embedding_model=custom_model)

                assert engine.embedding_model_name == custom_model

                # Trigger lazy loading
                _ = engine.embedding_model
                mock_transformer.assert_called_once_with(custom_model)

    def test_custom_collection_name(self):
        """Test initializing with a custom collection name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "custom_collection_test"

            with patch(
                "obsidian_mcp.vector_search.chromadb.PersistentClient"
            ) as mock_client:
                mock_client_instance = Mock()
                mock_client.return_value = mock_client_instance

                custom_collection = "custom_collection"
                engine = VectorSearchEngine(
                    index_path, collection_name=custom_collection
                )

                assert engine.collection_name == custom_collection
                mock_client_instance.get_or_create_collection.assert_called_once_with(
                    name=custom_collection, metadata={"hnsw:space": "cosine"}
                )
