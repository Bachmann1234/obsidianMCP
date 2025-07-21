"""File system watcher for incremental index updates."""

import logging
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .parser import ObsidianParser
from .search import ObsidianSearchIndex

logger = logging.getLogger(__name__)


class VaultWatcher(FileSystemEventHandler):
    """Watches for changes in the Obsidian vault and updates the search index."""
    
    def __init__(
        self,
        vault_path: Path,
        parser: ObsidianParser,
        search_index: ObsidianSearchIndex,
        on_change_callback: Optional[Callable] = None
    ):
        """Initialize the vault watcher."""
        self.vault_path = vault_path
        self.parser = parser
        self.search_index = search_index
        self.on_change_callback = on_change_callback
        self.observer: Optional[Observer] = None
    
    def start_watching(self) -> None:
        """Start watching the vault for changes."""
        if self.observer is not None:
            logger.warning("Watcher is already running")
            return
        
        self.observer = Observer()
        self.observer.schedule(self, str(self.vault_path), recursive=True)
        self.observer.start()
        logger.info(f"Started watching vault at {self.vault_path}")
    
    def stop_watching(self) -> None:
        """Stop watching the vault."""
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Stopped watching vault")
    
    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if self._should_process_file(file_path):
            logger.info(f"File created: {file_path}")
            self._update_note(file_path)
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if self._should_process_file(file_path):
            logger.info(f"File modified: {file_path}")
            self._update_note(file_path)
    
    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if self._should_process_file(file_path):
            logger.info(f"File deleted: {file_path}")
            self.search_index.remove_note(file_path)
            
            if self.on_change_callback:
                self.on_change_callback('deleted', file_path)
    
    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move/rename events."""
        if event.is_directory:
            return
        
        old_path = Path(event.src_path)
        new_path = Path(event.dest_path)
        
        if self._should_process_file(old_path):
            logger.info(f"File moved: {old_path} -> {new_path}")
            
            # Remove old file from index
            self.search_index.remove_note(old_path)
            
            # Add new file to index if it's still a markdown file
            if self._should_process_file(new_path):
                self._update_note(new_path)
            
            if self.on_change_callback:
                self.on_change_callback('moved', old_path, new_path)
    
    def _should_process_file(self, file_path: Path) -> bool:
        """Check if a file should be processed (is a markdown file and not in .obsidian)."""
        if file_path.suffix.lower() != '.md':
            return False
        
        if '.obsidian' in file_path.parts:
            return False
        
        return True
    
    def _update_note(self, file_path: Path) -> None:
        """Parse and update a note in the search index."""
        try:
            note = self.parser.parse_note(file_path)
            if note:
                self.search_index.add_note(note)
                
                if self.on_change_callback:
                    self.on_change_callback('updated', file_path, note)
            else:
                logger.warning(f"Failed to parse note: {file_path}")
        
        except Exception as e:
            logger.error(f"Error updating note {file_path}: {e}")


class VaultWatcherManager:
    """Manages the vault watcher lifecycle."""
    
    def __init__(
        self,
        vault_path: Path,
        parser: ObsidianParser,
        search_index: ObsidianSearchIndex,
        enabled: bool = True
    ):
        """Initialize the watcher manager."""
        self.vault_path = vault_path
        self.parser = parser
        self.search_index = search_index
        self.enabled = enabled
        self.watcher: Optional[VaultWatcher] = None
        self.stats = {
            'files_updated': 0,
            'files_deleted': 0,
            'files_moved': 0,
            'errors': 0
        }
    
    def start(self) -> None:
        """Start the vault watcher if enabled."""
        if not self.enabled:
            logger.info("File watching is disabled")
            return
        
        if self.watcher is not None:
            logger.warning("Watcher is already running")
            return
        
        self.watcher = VaultWatcher(
            vault_path=self.vault_path,
            parser=self.parser,
            search_index=self.search_index,
            on_change_callback=self._on_change
        )
        
        try:
            self.watcher.start_watching()
            logger.info("Vault watcher started successfully")
        except Exception as e:
            logger.error(f"Failed to start vault watcher: {e}")
            self.watcher = None
            raise
    
    def stop(self) -> None:
        """Stop the vault watcher."""
        if self.watcher is not None:
            self.watcher.stop_watching()
            self.watcher = None
            logger.info("Vault watcher stopped")
    
    def _on_change(self, action: str, *args) -> None:
        """Handle file change events and update statistics."""
        if action == 'updated':
            self.stats['files_updated'] += 1
        elif action == 'deleted':
            self.stats['files_deleted'] += 1
        elif action == 'moved':
            self.stats['files_moved'] += 1
        
        logger.debug(f"Vault change: {action} - {args}")
    
    def get_stats(self) -> dict:
        """Get watcher statistics."""
        return {
            'enabled': self.enabled,
            'running': self.watcher is not None,
            **self.stats
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()