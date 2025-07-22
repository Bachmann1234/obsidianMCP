"""Vector search functionality using ChromaDB and SentenceTransformers."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .parser import ObsidianNote

logger = logging.getLogger(__name__)


class VectorSearchEngine:
    """Vector search engine using ChromaDB and SentenceTransformers."""

    def __init__(
        self,
        index_path: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "obsidian_notes",
    ):
        """Initialize the vector search engine.

        Args:
            index_path: Path to store ChromaDB index
            embedding_model: SentenceTransformers model name
            collection_name: ChromaDB collection name
        """
        self.index_path = index_path
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name

        # Initialize ChromaDB client
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.index_path), settings=Settings(anonymized_telemetry=False)
        )

        # Initialize SentenceTransformers model (lazy loading)
        self._embedding_model: Optional[SentenceTransformer] = None

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Initialized vector search with model: {embedding_model}")

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def _generate_doc_id(self, note_path: str) -> str:
        """Generate a unique document ID from note path."""
        path_str = str(note_path) if hasattr(note_path, "__fspath__") else note_path
        return hashlib.md5(path_str.encode()).hexdigest()

    def _prepare_text_for_embedding(self, note: ObsidianNote) -> str:
        """Prepare note text for embedding generation."""
        # Combine title and content for richer embeddings
        text_parts = []

        if note.title:
            text_parts.append(f"Title: {note.title}")

        if note.content:
            # Clean up content - remove excessive whitespace
            content = " ".join(note.content.split())
            text_parts.append(f"Content: {content}")

        if note.tags:
            # Include tags for better semantic understanding
            tags_str = " ".join(note.tags)
            text_parts.append(f"Tags: {tags_str}")

        return " ".join(text_parts)

    def add_note(self, note: ObsidianNote) -> None:
        """Add or update a note in the vector index."""
        try:
            doc_id = self._generate_doc_id(note.path)
            text = self._prepare_text_for_embedding(note)

            # Generate embedding
            embedding = self.embedding_model.encode(text).tolist()

            # Prepare metadata
            metadata = {
                "path": str(note.path),
                "title": note.title or "",
                "tags": json.dumps(list(note.tags) if note.tags else []),
                "created_date": (
                    note.created_date.isoformat() if note.created_date else ""
                ),
                "modified_date": (
                    note.modified_date.isoformat() if note.modified_date else ""
                ),
                "content_length": len(note.content or ""),
            }

            # Check if document already exists
            existing = self.collection.get(ids=[doc_id])

            if existing["ids"]:
                # Update existing document
                self.collection.update(
                    ids=[doc_id],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[text],
                )
                logger.debug(f"Updated vector for note: {note.path}")
            else:
                # Add new document
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[text],
                )
                logger.debug(f"Added vector for note: {note.path}")

        except Exception as e:
            logger.error(f"Error adding note to vector index: {note.path}: {e}")

    def remove_note(self, note_path: str) -> None:
        """Remove a note from the vector index."""
        try:
            doc_id = self._generate_doc_id(note_path)

            # Check if document exists before trying to delete
            existing = self.collection.get(ids=[doc_id])
            if existing["ids"]:
                self.collection.delete(ids=[doc_id])
                logger.debug(f"Removed vector for note: {note_path}")

        except Exception as e:
            logger.error(f"Error removing note from vector index: {note_path}: {e}")

    def search(
        self, query: str, top_k: int = 10, tag_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar notes using vector similarity.

        Args:
            query: Search query text
            top_k: Number of results to return
            tag_filter: Optional list of tags to filter by

        Returns:
            List of search results with metadata and scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()

            # Prepare where clause for tag filtering
            where_clause = None
            if tag_filter:
                # Note: This is a simplified tag filter - ChromaDB doesn't support
                # complex JSON queries easily, so we'll filter post-search
                pass

            # Perform vector search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, 100),  # Get more results for filtering
                where=where_clause,
            )

            # Process and filter results
            processed_results = []
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]

                # Convert distance to similarity score (cosine similarity)
                similarity_score = 1.0 - distance

                # Apply tag filtering if specified
                if tag_filter:
                    note_tags = json.loads(metadata.get("tags", "[]"))
                    if not any(tag in note_tags for tag in tag_filter):
                        continue

                result = {
                    "path": metadata["path"],
                    "title": metadata["title"],
                    "similarity_score": similarity_score,
                    "tags": json.loads(metadata.get("tags", "[]")),
                    "created_date": metadata.get("created_date", ""),
                    "modified_date": metadata.get("modified_date", ""),
                    "content_length": metadata.get("content_length", 0),
                }

                processed_results.append(result)

                # Stop when we have enough results
                if len(processed_results) >= top_k:
                    break

            logger.debug(
                f"Vector search returned {len(processed_results)} results for query: {query}"
            )
            return processed_results

        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            return []

    def get_similar_notes(
        self, note_path: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Find notes similar to a given note.

        Args:
            note_path: Path of the reference note
            top_k: Number of similar notes to return

        Returns:
            List of similar notes with metadata and scores
        """
        try:
            doc_id = self._generate_doc_id(note_path)

            # Get the note's embedding
            existing = self.collection.get(
                ids=[doc_id], include=["embeddings", "metadatas"]
            )

            if not existing["ids"]:
                logger.warning(f"Note not found in vector index: {note_path}")
                return []

            embedding = existing["embeddings"][0]

            # Search for similar notes
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k + 1,  # +1 to exclude the query note itself
            )

            # Process results (exclude the query note itself)
            processed_results = []
            for i, result_id in enumerate(results["ids"][0]):
                if result_id == doc_id:
                    continue  # Skip the query note itself

                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                similarity_score = 1.0 - distance

                result = {
                    "path": metadata["path"],
                    "title": metadata["title"],
                    "similarity_score": similarity_score,
                    "tags": json.loads(metadata.get("tags", "[]")),
                    "created_date": metadata.get("created_date", ""),
                    "modified_date": metadata.get("modified_date", ""),
                }

                processed_results.append(result)

                if len(processed_results) >= top_k:
                    break

            return processed_results

        except Exception as e:
            logger.error(f"Error finding similar notes for {note_path}: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index."""
        try:
            count = self.collection.count()
            return {
                "total_notes": count,
                "embedding_model": self.embedding_model_name,
                "index_path": str(self.index_path),
                "collection_name": self.collection_name,
            }
        except Exception as e:
            logger.error(f"Error getting vector index stats: {e}")
            return {
                "total_notes": 0,
                "embedding_model": self.embedding_model_name,
                "index_path": str(self.index_path),
                "collection_name": self.collection_name,
                "error": str(e),
            }

    def rebuild_index(self, notes: List[ObsidianNote]) -> None:
        """Rebuild the entire vector index from scratch.

        Args:
            notes: List of all notes to index
        """
        logger.info("Rebuilding vector index...")

        try:
            # Clear existing collection
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )

            # Add all notes in batches for efficiency
            batch_size = 100
            for i in range(0, len(notes), batch_size):
                batch = notes[i : i + batch_size]
                for note in batch:
                    self.add_note(note)

                logger.info(
                    f"Processed {min(i + batch_size, len(notes))}/{len(notes)} notes"
                )

            logger.info(f"Vector index rebuild complete. Indexed {len(notes)} notes.")

        except Exception as e:
            logger.error(f"Error rebuilding vector index: {e}")
            raise
