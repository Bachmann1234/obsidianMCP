"""Markdown and frontmatter parsing for Obsidian notes."""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import frontmatter
from pydantic import BaseModel, ConfigDict


class ObsidianNote(BaseModel):
    """Represents a parsed Obsidian note."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    path: Path
    title: str
    content: str
    raw_content: str
    frontmatter: Dict
    tags: Set[str]
    wikilinks: Set[str]
    backlinks: Set[str]
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None


class ObsidianParser:
    """Parser for Obsidian markdown files."""
    
    # Regex patterns for Obsidian syntax
    WIKILINK_PATTERN = re.compile(r'\[\[([^\]]+)\]\]')
    TAG_PATTERN = re.compile(r'(?:^|\s)#([a-zA-Z0-9/_-]+)')
    YAML_TAG_PATTERN = re.compile(r'^tags:\s*(.+)$', re.MULTILINE)
    
    def __init__(self, vault_path: Path):
        """Initialize parser with vault path."""
        self.vault_path = vault_path
        self._note_cache: Dict[Path, ObsidianNote] = {}
    
    def parse_note(self, file_path: Path) -> Optional[ObsidianNote]:
        """Parse a single markdown file into an ObsidianNote."""
        if not file_path.suffix.lower() == '.md':
            return None
        
        try:
            # Read and parse frontmatter
            with open(file_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
            
            raw_content = file_path.read_text(encoding='utf-8')
            content = post.content
            metadata = post.metadata
            
            # Extract title
            title = self._extract_title(file_path, content, metadata)
            
            # Extract tags
            tags = self._extract_tags(content, metadata)
            
            # Extract wikilinks
            wikilinks = self._extract_wikilinks(content)
            
            # Get file timestamps
            stat = file_path.stat()
            modified_date = datetime.fromtimestamp(stat.st_mtime)
            created_date = datetime.fromtimestamp(stat.st_ctime)
            
            note = ObsidianNote(
                path=file_path,
                title=title,
                content=content,
                raw_content=raw_content,
                frontmatter=metadata,
                tags=tags,
                wikilinks=wikilinks,
                backlinks=set(),  # Will be populated separately
                created_date=created_date,
                modified_date=modified_date
            )
            
            self._note_cache[file_path] = note
            return note
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def _extract_title(self, file_path: Path, content: str, metadata: Dict) -> str:
        """Extract note title from frontmatter, first heading, or filename."""
        # Check frontmatter for title
        if 'title' in metadata:
            return metadata['title']
        
        # Check for first H1 heading
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        
        # Fall back to filename without extension
        return file_path.stem
    
    def _extract_tags(self, content: str, metadata: Dict) -> Set[str]:
        """Extract tags from frontmatter and inline tags."""
        tags = set()
        
        # Extract from frontmatter
        if 'tags' in metadata:
            frontmatter_tags = metadata['tags']
            if isinstance(frontmatter_tags, str):
                # Handle comma-separated tags
                tags.update(tag.strip() for tag in frontmatter_tags.split(','))
            elif isinstance(frontmatter_tags, list):
                tags.update(str(tag).strip() for tag in frontmatter_tags)
        
        # Extract inline tags (#tag format)
        inline_tags = self.TAG_PATTERN.findall(content)
        tags.update(inline_tags)
        
        return tags
    
    def _extract_wikilinks(self, content: str) -> Set[str]:
        """Extract wikilinks from content."""
        wikilinks = set()
        matches = self.WIKILINK_PATTERN.findall(content)
        
        for match in matches:
            # Handle [[Note Name|Display Text]] format
            link_target = match.split('|')[0].strip()
            wikilinks.add(link_target)
        
        return wikilinks
    
    def discover_notes(self) -> List[Path]:
        """Discover all markdown files in the vault."""
        notes = []
        for file_path in self.vault_path.rglob('*.md'):
            # Skip files in .obsidian directory
            if '.obsidian' in file_path.parts:
                continue
            notes.append(file_path)
        
        return sorted(notes)
    
    def compute_backlinks(self, notes: List[ObsidianNote]) -> None:
        """Compute backlinks for all notes."""
        # Create a mapping from note names to notes
        note_map = {}
        for note in notes:
            # Add both the filename (with and without extension) and title
            note_map[note.path.stem] = note
            note_map[note.title] = note
        
        # Compute backlinks
        for note in notes:
            for wikilink in note.wikilinks:
                if wikilink in note_map:
                    target_note = note_map[wikilink]
                    target_note.backlinks.add(note.path.stem)
    
    def get_note_by_name(self, name: str) -> Optional[ObsidianNote]:
        """Get a note by its name (filename or title)."""
        for note in self._note_cache.values():
            if note.path.stem == name or note.title == name:
                return note
        return None
    
    def clear_cache(self) -> None:
        """Clear the note cache."""
        self._note_cache.clear()