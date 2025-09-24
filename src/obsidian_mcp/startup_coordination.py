"""Process-level startup coordination for MCP server instances."""

import atexit
import logging
import os
import time
from pathlib import Path
from typing import IO, Any, Optional

import portalocker

logger = logging.getLogger(__name__)


class StartupCoordinator:
    """Coordinates startup between multiple MCP server processes.

    Prevents multiple instances from initializing the search index simultaneously
    by using process-level file locking.
    """

    def __init__(self, index_path: Path, timeout: float = 60.0):
        """Initialize startup coordinator.

        Args:
            index_path: Path to the search index directory
            timeout: Maximum time to wait for coordination lock (seconds)
        """
        self.index_path = index_path
        self.timeout = timeout
        self.lock_file_path = index_path / ".startup.lock"
        self._lock_file: Optional[IO[Any]] = None
        self._is_coordinator = False

        # Ensure index directory exists
        index_path.mkdir(parents=True, exist_ok=True)

        # Register cleanup on exit
        atexit.register(self._cleanup)

    def acquire_coordination_lock(self) -> bool:
        """Acquire the startup coordination lock.

        Returns:
            True if this process should coordinate startup (initialize index),
            False if another process is already coordinating or coordination failed.
        """
        try:
            logger.info(
                f"Attempting to acquire startup coordination lock: {self.lock_file_path}"
            )

            # First, do a quick check if initialization appears to already be complete
            if self._check_initialization_complete():
                logger.info(
                    "Initialization appears to already be complete, skipping lock acquisition"
                )
                return False

            # Try to acquire exclusive lock
            self._lock_file = open(self.lock_file_path, "w")

            try:
                # Try non-blocking first
                portalocker.lock(
                    self._lock_file, portalocker.LOCK_EX | portalocker.LOCK_NB
                )
                self._is_coordinator = True

                # Write process info to lock file for debugging
                self._lock_file.write(f"pid:{os.getpid()}\ntime:{time.time()}\n")
                self._lock_file.flush()

                logger.info(
                    f"Successfully acquired startup coordination lock (PID: {os.getpid()})"
                )
                return True

            except portalocker.LockException:
                # Lock is held by another process - we should wait, not coordinate
                logger.info("Startup coordination lock held by another process")
                self._cleanup()
                return False

        except Exception as e:
            logger.error(f"Error acquiring startup coordination lock: {e}")
            self._cleanup()
            return False

    def wait_for_initialization(self, check_interval: float = 0.5) -> bool:
        """Wait for another process to complete initialization.

        Args:
            check_interval: How often to check if initialization is complete (seconds)

        Returns:
            True if initialization was completed successfully, False on timeout or error.
        """
        logger.info("Waiting for another process to complete initialization...")

        start_time = time.time()
        while time.time() - start_time < self.timeout:
            # Check if the lock file still exists (coordinator process is still working)
            if not self.lock_file_path.exists():
                logger.info(
                    "Startup coordination lock released, checking if initialization completed..."
                )
                return self._check_initialization_complete()

            # Wait before checking again
            time.sleep(check_interval)

        logger.warning(f"Timed out waiting for initialization after {self.timeout}s")
        return False

    def release_coordination_lock(self) -> None:
        """Release the startup coordination lock."""
        if self._is_coordinator and self._lock_file:
            try:
                logger.info("Releasing startup coordination lock")
                portalocker.unlock(self._lock_file)
                self._lock_file.close()

                # Remove lock file to signal completion
                if self.lock_file_path.exists():
                    self.lock_file_path.unlink()

                logger.info("Startup coordination lock released successfully")

            except Exception as e:
                logger.error(f"Error releasing startup coordination lock: {e}")
            finally:
                self._lock_file = None
                self._is_coordinator = False

    def _check_initialization_complete(self) -> bool:
        """Check if index initialization appears to be complete.

        Returns:
            True if initialization seems complete, False otherwise.
        """
        try:
            # Check for index files that indicate successful initialization
            whoosh_files = list(
                self.index_path.glob("*.toc")
            )  # Whoosh table of contents files
            segment_files = list(self.index_path.glob("_*.seg"))  # Whoosh segment files

            if not (whoosh_files or segment_files):
                logger.warning("Index initialization does not appear to be complete")
                return False

            # Also check for actual index content, not just files
            try:
                from whoosh import index

                if index.exists_in(str(self.index_path)):
                    idx = index.open_dir(str(self.index_path))
                    with idx.searcher() as searcher:
                        doc_count = searcher.doc_count()
                        if doc_count > 0:
                            logger.info(
                                f"Index initialization appears to be complete ({doc_count} documents)"
                            )
                            return True
                        else:
                            logger.info("Index files exist but index appears empty")
                            return False
                else:
                    logger.info("Index files present but no valid Whoosh index found")
                    return False
            except Exception as e:
                logger.info(f"Cannot verify index content: {e}")
                # Fall back to file-based check if we can't read the index
                logger.info(
                    "Index initialization appears to be complete (file-based check)"
                )
                return True

        except Exception as e:
            logger.error(f"Error checking initialization status: {e}")
            return False

    def _cleanup(self) -> None:
        """Clean up resources and lock file."""
        try:
            if self._is_coordinator and self._lock_file:
                # Release the lock
                try:
                    portalocker.unlock(self._lock_file)
                except:
                    pass  # May already be unlocked

                try:
                    self._lock_file.close()
                except:
                    pass  # May already be closed

                # Remove lock file
                try:
                    if self.lock_file_path.exists():
                        self.lock_file_path.unlink()
                        logger.debug("Cleanup: removed startup coordination lock file")
                except Exception as e:
                    logger.debug(f"Cleanup: could not remove lock file: {e}")

        except Exception as e:
            logger.debug(f"Error during startup coordination cleanup: {e}")
        finally:
            self._lock_file = None
            self._is_coordinator = False

    def is_coordinator(self) -> bool:
        """Check if this process is the coordinator (holds the lock).

        Returns:
            True if this process is coordinating startup, False otherwise.
        """
        return self._is_coordinator

    def __enter__(self) -> "StartupCoordinator":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.release_coordination_lock()
