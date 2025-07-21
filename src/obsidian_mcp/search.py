"""Whoosh-based search index for Obsidian notes."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from whoosh import fields, index
from whoosh.filedb.filestore import FileStorage
from whoosh.qparser import MultifieldParser, QueryParser
from whoosh.query import Query
from whoosh.searching import Results
from whoosh.writing import IndexWriter

from .parser import ObsidianNote

logger = logging.getLogger(__name__)


class SearchResult:
    """Represents a search result."""
    
    def __init__(self, note: ObsidianNote, score: float, highlights: Dict[str, str]):
        self.note = note
        self.score = score
        self.highlights = highlights


class ObsidianSearchIndex:
    """Whoosh-based search index for Obsidian notes."""
    
    # Index schema
    SCHEMA = fields.Schema(
        path=fields.ID(stored=True, unique=True),
        title=fields.TEXT(stored=True, phrase=True),
        content=fields.TEXT(stored=True),
        tags=fields.KEYWORD(stored=True, commas=True),
        wikilinks=fields.KEYWORD(stored=True, commas=True),
        created_date=fields.DATETIME(stored=True),
        modified_date=fields.DATETIME(stored=True),
        frontmatter=fields.TEXT(stored=True)
    )
    
    def __init__(self, index_path: Path):
        """Initialize search index."""
        self.index_path = index_path
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._index: Optional[index.Index] = None
        self._ensure_index()
    
    def _ensure_index(self) -> None:
        """Ensure the index exists and is properly initialized."""
        if index.exists_in(str(self.index_path)):
            self._index = index.open_dir(str(self.index_path))
        else:
            self._index = index.create_in(str(self.index_path), self.SCHEMA)
    
    def add_note(self, note: ObsidianNote) -> None:
        """Add or update a note in the index."""
        with self._index.writer() as writer:
            writer.update_document(
                path=str(note.path),
                title=note.title,
                content=note.content,
                tags=','.join(note.tags),
                wikilinks=','.join(note.wikilinks),
                created_date=note.created_date,
                modified_date=note.modified_date,
                frontmatter=str(note.frontmatter)
            )
    
    def remove_note(self, file_path: Path) -> None:
        """Remove a note from the index."""
        with self._index.writer() as writer:
            writer.delete_by_term('path', str(file_path))
    
    def bulk_add_notes(self, notes: List[ObsidianNote]) -> None:
        """Add multiple notes to the index efficiently."""
        with self._index.writer() as writer:
            for note in notes:
                writer.update_document(
                    path=str(note.path),
                    title=note.title,
                    content=note.content,
                    tags=','.join(note.tags),
                    wikilinks=','.join(note.wikilinks),
                    created_date=note.created_date,
                    modified_date=note.modified_date,
                    frontmatter=str(note.frontmatter)
                )
    
    def search(
        self,
        query: str,
        limit: int = 50,
        tags: Optional[Set[str]] = None,
        search_fields: Optional[List[str]] = None
    ) -> List[Dict]:
        """Search the index and return results."""
        if not query.strip():
            return []
        
        # Default search fields
        if search_fields is None:
            search_fields = ['title', 'content', 'tags']
        
        with self._index.searcher() as searcher:
            # Create parser for multi-field search
            parser = MultifieldParser(search_fields, self._index.schema)
            
            # Parse the query
            try:
                parsed_query = parser.parse(query)
            except Exception:
                # Fall back to simple content search if parsing fails
                parser = QueryParser('content', self._index.schema)
                parsed_query = parser.parse(query)
            
            # Add tag filter if specified
            if tags:
                tag_queries = []
                for tag in tags:
                    tag_parser = QueryParser('tags', self._index.schema)
                    tag_queries.append(tag_parser.parse(tag))
                
                if tag_queries:
                    from whoosh.query import And, Or
                    tag_query = Or(tag_queries) if len(tag_queries) > 1 else tag_queries[0]
                    parsed_query = And([parsed_query, tag_query])
            
            # Execute search
            results = searcher.search(parsed_query, limit=limit)
            
            # Convert to our format
            search_results = []
            for result in results:
                # Get highlights - this is a method that needs to be called
                highlights = {}
                try:
                    if hasattr(result, 'highlights'):
                        highlights = dict(result.highlights())
                except:
                    highlights = {}
                
                search_results.append({
                    'path': result['path'],
                    'title': result['title'],
                    'content': result['content'][:500] + '...' if len(result['content']) > 500 else result['content'],
                    'tags': result['tags'].split(',') if result['tags'] else [],
                    'score': result.score,
                    'highlights': highlights,
                    'created_date': result['created_date'].isoformat() if result['created_date'] else None,
                    'modified_date': result['modified_date'].isoformat() if result['modified_date'] else None
                })
            
            return search_results
    
    def get_note_by_path(self, file_path: Path) -> Optional[Dict]:
        """Get a specific note by its path."""
        with self._index.searcher() as searcher:
            results = searcher.documents(path=str(file_path))
            for result in results:
                return {
                    'path': result['path'],
                    'title': result['title'],
                    'content': result['content'],
                    'tags': result['tags'].split(',') if result['tags'] else [],
                    'wikilinks': result['wikilinks'].split(',') if result['wikilinks'] else [],
                    'created_date': result['created_date'].isoformat() if result['created_date'] else None,
                    'modified_date': result['modified_date'].isoformat() if result['modified_date'] else None
                }
        return None
    
    def list_all_tags(self) -> List[str]:
        """Get all unique tags in the index."""
        tags = set()
        with self._index.searcher() as searcher:
            for fields in searcher.all_stored_fields():
                if fields.get('tags'):
                    note_tags = fields['tags'].split(',')
                    tags.update(tag.strip() for tag in note_tags if tag.strip())
        return sorted(list(tags))
    
    def get_recent_notes(self, limit: int = 10) -> List[Dict]:
        """Get recently modified notes."""
        with self._index.searcher() as searcher:
            results = searcher.documents()
            # Sort by modified date
            sorted_results = sorted(
                results,
                key=lambda x: x.get('modified_date', datetime.min),
                reverse=True
            )
            
            recent_notes = []
            for result in sorted_results[:limit]:
                recent_notes.append({
                    'path': result['path'],
                    'title': result['title'],
                    'content': result['content'][:200] + '...' if len(result['content']) > 200 else result['content'],
                    'tags': result['tags'].split(',') if result['tags'] else [],
                    'modified_date': result['modified_date'].isoformat() if result['modified_date'] else None
                })
            
            return recent_notes
    
    def rebuild_index(self, notes: List[ObsidianNote]) -> None:
        """Completely rebuild the search index."""
        # Clear existing index by removing all documents
        with self._index.writer() as writer:
            # Truncate the index (remove all documents)
            from whoosh.query import Every
            writer.delete_by_query(Every())
        
        # Add all notes
        self.bulk_add_notes(notes)
    
    def optimize_index(self) -> None:
        """Optimize the index for better performance."""
        with self._index.writer() as writer:
            writer.commit(optimize=True)
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        with self._index.searcher() as searcher:
            return {
                'doc_count': searcher.doc_count(),
                'field_names': list(self._index.schema.names()),
                'index_path': str(self.index_path),
                'last_modified': self._get_index_last_modified()
            }
    
    def _get_index_last_modified(self) -> Optional[str]:
        """Get the last modification time of the index."""
        try:
            # Check modification time of the main index file
            index_files = list(self.index_path.glob("*"))
            if index_files:
                latest_mtime = max(f.stat().st_mtime for f in index_files)
                return datetime.fromtimestamp(latest_mtime).isoformat()
        except Exception:
            pass
        return None
    
    def needs_update(self, vault_path: Path) -> bool:
        """Check if index needs updating based on file modification times."""
        try:
            # Get index last modified time
            index_mtime = self._get_index_last_modified()
            if not index_mtime:
                return True  # No index exists
            
            index_dt = datetime.fromisoformat(index_mtime)
            
            # Check if any markdown files are newer than the index
            for md_file in vault_path.rglob('*.md'):
                if '.obsidian' in md_file.parts:
                    continue
                
                file_mtime = datetime.fromtimestamp(md_file.stat().st_mtime)
                if file_mtime > index_dt:
                    return True
            
            return False
        except Exception:
            return True  # Err on the side of caution
    
    def incremental_update(self, vault_path: Path, parser) -> Dict[str, int]:
        """Perform incremental update of files newer than index."""
        stats = {'updated': 0, 'added': 0, 'removed': 0}
        
        try:
            index_mtime_str = self._get_index_last_modified()
            if not index_mtime_str:
                return stats
            
            index_dt = datetime.fromisoformat(index_mtime_str)
            
            # Find files that need updating
            files_to_update = []
            for md_file in vault_path.rglob('*.md'):
                if '.obsidian' in md_file.parts:
                    continue
                
                file_mtime = datetime.fromtimestamp(md_file.stat().st_mtime)
                if file_mtime > index_dt:
                    files_to_update.append(md_file)
            
            # Update modified files
            for file_path in files_to_update:
                note = parser.parse_note(file_path)
                if note:
                    # Check if note already exists in index
                    existing = self.get_note_by_path(file_path)
                    if existing:
                        stats['updated'] += 1
                    else:
                        stats['added'] += 1
                    
                    self.add_note(note)
            
            # TODO: Handle removed files (would need to track indexed files)
            
        except Exception as e:
            logger.error(f"Error during incremental update: {e}")
        
        return stats