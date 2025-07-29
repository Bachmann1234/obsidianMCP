"""Tests for the file watcher module."""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from obsidian_mcp.parser import ObsidianParser
from obsidian_mcp.search import ObsidianSearchIndex
from obsidian_mcp.watcher import VaultWatcher, VaultWatcherManager


@pytest.fixture
def temp_vault():
    """Create a temporary vault for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir)
        yield vault_path


@pytest.fixture
def temp_index():
    """Create a temporary search index."""
    with tempfile.TemporaryDirectory() as temp_dir:
        index_path = Path(temp_dir) / "test_index"
        yield ObsidianSearchIndex(index_path)


@pytest.fixture
def mock_parser():
    """Create a mock parser."""
    parser = Mock(spec=ObsidianParser)
    return parser


@pytest.fixture
def mock_search_index():
    """Create a mock search index."""
    search_index = Mock(spec=ObsidianSearchIndex)
    return search_index


def test_vault_watcher_initialization(temp_vault, mock_parser, mock_search_index):
    """Test VaultWatcher initialization."""
    callback = Mock()

    watcher = VaultWatcher(
        vault_path=temp_vault,
        parser=mock_parser,
        search_index=mock_search_index,
        on_change_callback=callback,
    )

    assert watcher.vault_path == temp_vault
    assert watcher.parser == mock_parser
    assert watcher.search_index == mock_search_index
    assert watcher.on_change_callback == callback
    assert watcher.observer is None


def test_vault_watcher_should_process_file(temp_vault, mock_parser, mock_search_index):
    """Test file filtering logic."""
    watcher = VaultWatcher(temp_vault, mock_parser, mock_search_index)

    # Should process markdown files
    assert watcher._should_process_file(Path("note.md"))
    assert watcher._should_process_file(Path("Note.MD"))

    # Should not process non-markdown files
    assert not watcher._should_process_file(Path("image.png"))
    assert not watcher._should_process_file(Path("document.txt"))
    assert not watcher._should_process_file(Path("data.json"))

    # Should not process files in .obsidian directory
    assert not watcher._should_process_file(Path(".obsidian/config.md"))
    assert not watcher._should_process_file(Path("vault/.obsidian/workspace.md"))


def test_vault_watcher_update_note_success(temp_vault, mock_parser, mock_search_index):
    """Test successful note update."""
    mock_note = Mock()
    mock_parser.parse_note.return_value = mock_note
    callback = Mock()

    watcher = VaultWatcher(temp_vault, mock_parser, mock_search_index, callback)

    test_file = temp_vault / "test.md"
    watcher._update_note(test_file)

    mock_parser.parse_note.assert_called_once_with(test_file)
    mock_search_index.add_note.assert_called_once_with(mock_note)
    callback.assert_called_once_with("updated", test_file, mock_note)


def test_vault_watcher_update_note_parse_failure(
    temp_vault, mock_parser, mock_search_index
):
    """Test note update when parsing fails."""
    mock_parser.parse_note.return_value = None
    callback = Mock()

    watcher = VaultWatcher(temp_vault, mock_parser, mock_search_index, callback)

    test_file = temp_vault / "test.md"
    watcher._update_note(test_file)

    mock_parser.parse_note.assert_called_once_with(test_file)
    mock_search_index.add_note.assert_not_called()
    callback.assert_not_called()


def test_vault_watcher_update_note_exception(
    temp_vault, mock_parser, mock_search_index
):
    """Test note update when an exception occurs."""
    mock_parser.parse_note.side_effect = Exception("Parse error")
    callback = Mock()

    watcher = VaultWatcher(temp_vault, mock_parser, mock_search_index, callback)

    test_file = temp_vault / "test.md"
    # Should not raise exception
    watcher._update_note(test_file)

    mock_parser.parse_note.assert_called_once_with(test_file)
    mock_search_index.add_note.assert_not_called()
    callback.assert_not_called()


def test_vault_watcher_on_created(temp_vault, mock_parser, mock_search_index):
    """Test file creation event handling."""
    mock_event = Mock()
    mock_event.is_directory = False
    mock_event.src_path = str(temp_vault / "new_note.md")

    watcher = VaultWatcher(temp_vault, mock_parser, mock_search_index)

    with patch.object(watcher, "_update_note") as mock_update:
        watcher.on_created(mock_event)
        mock_update.assert_called_once_with(Path(mock_event.src_path))


def test_vault_watcher_on_created_directory(temp_vault, mock_parser, mock_search_index):
    """Test that directory creation events are ignored."""
    mock_event = Mock()
    mock_event.is_directory = True
    mock_event.src_path = str(temp_vault / "new_folder")

    watcher = VaultWatcher(temp_vault, mock_parser, mock_search_index)

    with patch.object(watcher, "_update_note") as mock_update:
        watcher.on_created(mock_event)
        mock_update.assert_not_called()


def test_vault_watcher_on_modified(temp_vault, mock_parser, mock_search_index):
    """Test file modification event handling."""
    mock_event = Mock()
    mock_event.is_directory = False
    mock_event.src_path = str(temp_vault / "modified_note.md")

    watcher = VaultWatcher(temp_vault, mock_parser, mock_search_index)

    with patch.object(watcher, "_update_note") as mock_update:
        watcher.on_modified(mock_event)
        mock_update.assert_called_once_with(Path(mock_event.src_path))


def test_vault_watcher_on_deleted(temp_vault, mock_parser, mock_search_index):
    """Test file deletion event handling."""
    mock_event = Mock()
    mock_event.is_directory = False
    mock_event.src_path = str(temp_vault / "deleted_note.md")

    callback = Mock()
    watcher = VaultWatcher(temp_vault, mock_parser, mock_search_index, callback)

    watcher.on_deleted(mock_event)

    mock_search_index.remove_note.assert_called_once_with(Path(mock_event.src_path))
    callback.assert_called_once_with("deleted", Path(mock_event.src_path))


def test_vault_watcher_on_moved(temp_vault, mock_parser, mock_search_index):
    """Test file move/rename event handling."""
    mock_event = Mock()
    mock_event.is_directory = False
    mock_event.src_path = str(temp_vault / "old_note.md")
    mock_event.dest_path = str(temp_vault / "new_note.md")

    callback = Mock()
    watcher = VaultWatcher(temp_vault, mock_parser, mock_search_index, callback)

    with patch.object(watcher, "_update_note") as mock_update:
        watcher.on_moved(mock_event)

        # Should remove old file and add new file
        mock_search_index.remove_note.assert_called_once_with(Path(mock_event.src_path))
        mock_update.assert_called_once_with(Path(mock_event.dest_path))
        callback.assert_called_once_with(
            "moved", Path(mock_event.src_path), Path(mock_event.dest_path)
        )


def test_vault_watcher_on_moved_to_non_markdown(
    temp_vault, mock_parser, mock_search_index
):
    """Test file move to non-markdown file."""
    mock_event = Mock()
    mock_event.is_directory = False
    mock_event.src_path = str(temp_vault / "note.md")
    mock_event.dest_path = str(temp_vault / "note.txt")  # Not markdown

    callback = Mock()
    watcher = VaultWatcher(temp_vault, mock_parser, mock_search_index, callback)

    with patch.object(watcher, "_update_note") as mock_update:
        watcher.on_moved(mock_event)

        # Should only remove old file, not add new one
        mock_search_index.remove_note.assert_called_once_with(Path(mock_event.src_path))
        mock_update.assert_not_called()
        callback.assert_called_once_with(
            "moved", Path(mock_event.src_path), Path(mock_event.dest_path)
        )


def test_vault_watcher_manager_initialization(
    temp_vault, mock_parser, mock_search_index
):
    """Test VaultWatcherManager initialization."""
    manager = VaultWatcherManager(
        temp_vault, mock_parser, mock_search_index, enabled=True
    )

    assert manager.vault_path == temp_vault
    assert manager.parser == mock_parser
    assert manager.search_index == mock_search_index
    assert manager.enabled is True
    assert manager.watcher is None
    assert manager.stats == {
        "files_updated": 0,
        "files_deleted": 0,
        "files_moved": 0,
        "errors": 0,
    }


def test_vault_watcher_manager_disabled(temp_vault, mock_parser, mock_search_index):
    """Test VaultWatcherManager when disabled."""
    manager = VaultWatcherManager(
        temp_vault, mock_parser, mock_search_index, enabled=False
    )

    manager.start()
    assert manager.watcher is None


def test_vault_watcher_manager_start_success(
    temp_vault, mock_parser, mock_search_index
):
    """Test successful watcher start."""
    manager = VaultWatcherManager(
        temp_vault, mock_parser, mock_search_index, enabled=True
    )

    with patch("obsidian_mcp.watcher.VaultWatcher") as mock_watcher_class:
        mock_watcher_instance = Mock()
        mock_watcher_class.return_value = mock_watcher_instance

        manager.start()

        assert manager.watcher is not None
        mock_watcher_instance.start_watching.assert_called_once()


def test_vault_watcher_manager_start_twice(temp_vault, mock_parser, mock_search_index):
    """Test starting watcher twice (should warn but not fail)."""
    manager = VaultWatcherManager(
        temp_vault, mock_parser, mock_search_index, enabled=True
    )

    with patch("obsidian_mcp.watcher.VaultWatcher") as mock_watcher_class:
        mock_watcher_instance = Mock()
        mock_watcher_class.return_value = mock_watcher_instance

        manager.start()
        manager.start()  # Second start should be ignored

        # Should only create watcher once
        mock_watcher_class.assert_called_once()


def test_vault_watcher_manager_stop(temp_vault, mock_parser, mock_search_index):
    """Test stopping the watcher."""
    manager = VaultWatcherManager(
        temp_vault, mock_parser, mock_search_index, enabled=True
    )

    with patch("obsidian_mcp.watcher.VaultWatcher") as mock_watcher_class:
        mock_watcher_instance = Mock()
        mock_watcher_class.return_value = mock_watcher_instance

        manager.start()
        manager.stop()

        mock_watcher_instance.stop_watching.assert_called_once()
        assert manager.watcher is None


def test_vault_watcher_manager_stop_when_not_started(
    temp_vault, mock_parser, mock_search_index
):
    """Test stopping watcher when it was never started."""
    manager = VaultWatcherManager(
        temp_vault, mock_parser, mock_search_index, enabled=True
    )

    # Should not raise exception
    manager.stop()
    assert manager.watcher is None


def test_vault_watcher_manager_on_change_stats(
    temp_vault, mock_parser, mock_search_index
):
    """Test that change events update statistics."""
    manager = VaultWatcherManager(
        temp_vault, mock_parser, mock_search_index, enabled=True
    )

    # Test different action types
    manager._on_change("updated", Path("test.md"))
    assert manager.stats["files_updated"] == 1

    manager._on_change("deleted", Path("test.md"))
    assert manager.stats["files_deleted"] == 1

    manager._on_change("moved", Path("old.md"), Path("new.md"))
    assert manager.stats["files_moved"] == 1


def test_vault_watcher_manager_get_stats(temp_vault, mock_parser, mock_search_index):
    """Test getting watcher statistics."""
    manager = VaultWatcherManager(
        temp_vault, mock_parser, mock_search_index, enabled=True
    )

    stats = manager.get_stats()

    expected_stats = {
        "enabled": True,
        "running": False,  # Not started yet
        "files_updated": 0,
        "files_deleted": 0,
        "files_moved": 0,
        "errors": 0,
    }

    assert stats == expected_stats


def test_vault_watcher_manager_context_manager(
    temp_vault, mock_parser, mock_search_index
):
    """Test VaultWatcherManager as context manager."""
    manager = VaultWatcherManager(
        temp_vault, mock_parser, mock_search_index, enabled=True
    )

    with patch("obsidian_mcp.watcher.VaultWatcher") as mock_watcher_class:
        mock_watcher_instance = Mock()
        mock_watcher_class.return_value = mock_watcher_instance

        with manager:
            # Watcher should be started
            mock_watcher_instance.start_watching.assert_called_once()

        # Watcher should be stopped on exit
        mock_watcher_instance.stop_watching.assert_called_once()


def test_vault_watcher_start_stop_watching(temp_vault, mock_parser, mock_search_index):
    """Test start and stop watching methods."""
    watcher = VaultWatcher(temp_vault, mock_parser, mock_search_index)

    with patch("obsidian_mcp.watcher.Observer") as mock_observer_class:
        mock_observer = Mock()
        mock_observer_class.return_value = mock_observer

        # Test start watching
        watcher.start_watching()

        assert watcher.observer is mock_observer
        mock_observer.schedule.assert_called_once_with(
            watcher, str(temp_vault), recursive=True
        )
        mock_observer.start.assert_called_once()

        # Test stop watching
        watcher.stop_watching()

        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()
        assert watcher.observer is None


def test_vault_watcher_start_watching_twice(temp_vault, mock_parser, mock_search_index):
    """Test starting watcher twice should warn."""
    watcher = VaultWatcher(temp_vault, mock_parser, mock_search_index)

    with patch("obsidian_mcp.watcher.Observer") as mock_observer_class:
        mock_observer = Mock()
        mock_observer_class.return_value = mock_observer

        watcher.start_watching()
        watcher.start_watching()  # Second call should be ignored

        # Observer should only be created once
        mock_observer_class.assert_called_once()


def test_vault_watcher_stop_watching_when_not_started(
    temp_vault, mock_parser, mock_search_index
):
    """Test stopping watcher when not started."""
    watcher = VaultWatcher(temp_vault, mock_parser, mock_search_index)

    # Should not raise exception
    watcher.stop_watching()
    assert watcher.observer is None
