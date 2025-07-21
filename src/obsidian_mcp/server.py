"""MCP server implementation for Obsidian vault search."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import click
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ListToolsRequest,
    TextContent,
    Tool,
)
from mcp import McpError
from pydantic import BaseModel

from .config import ServerConfig, load_config_from_env
from .parser import ObsidianParser
from .search import ObsidianSearchIndex
from .watcher import VaultWatcherManager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObsidianMCPServer:
    """MCP Server for Obsidian vault search."""
    
    def __init__(self, config: ServerConfig):
        """Initialize the server with configuration."""
        self.config = config
        self.parser = ObsidianParser(config.vault_path)
        self.search_index = ObsidianSearchIndex(config.index_path)
        self.watcher = VaultWatcherManager(
            vault_path=config.vault_path,
            parser=self.parser,
            search_index=self.search_index,
            enabled=config.watch_for_changes
        )
        self.server = Server("obsidian-mcp-server")
        self._setup_tools()
        
        # Initialize index if needed
        if config.auto_rebuild_index:
            asyncio.create_task(self._initialize_index())
    
    def _setup_tools(self) -> None:
        """Setup MCP tools."""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="search_notes",
                    description="Search through Obsidian notes using full-text search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query string"
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filter by specific tags (optional)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10)",
                                "default": 10
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_note",
                    description="Get the full content of a specific note by path or title",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "identifier": {
                                "type": "string",
                                "description": "Note path (relative to vault) or note title"
                            }
                        },
                        "required": ["identifier"]
                    }
                ),
                Tool(
                    name="list_recent_notes",
                    description="Get recently modified notes",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of notes to return (default: 10)",
                                "default": 10
                            }
                        }
                    }
                ),
                Tool(
                    name="get_all_tags",
                    description="Get all available tags in the vault",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="search_by_tag",
                    description="Find all notes with specific tags",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Tags to search for"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 20)",
                                "default": 20
                            }
                        },
                        "required": ["tags"]
                    }
                ),
                Tool(
                    name="get_vault_stats",
                    description="Get statistics about the vault and search index",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "search_notes":
                    return await self._search_notes(arguments)
                elif name == "get_note":
                    return await self._get_note(arguments)
                elif name == "list_recent_notes":
                    return await self._list_recent_notes(arguments)
                elif name == "get_all_tags":
                    return await self._get_all_tags(arguments)
                elif name == "search_by_tag":
                    return await self._search_by_tag(arguments)
                elif name == "get_vault_stats":
                    return await self._get_vault_stats(arguments)
                else:
                    raise McpError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                raise McpError(f"Tool execution failed: {str(e)}")
    
    async def _search_notes(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Search notes using full-text search."""
        query = arguments.get("query", "")
        tags = set(arguments.get("tags", []))
        limit = min(arguments.get("limit", 10), self.config.max_results)
        
        if not query.strip():
            return [TextContent(type="text", text="Empty search query")]
        
        results = self.search_index.search(
            query=query,
            limit=limit,
            tags=tags if tags else None
        )
        
        if not results:
            return [TextContent(type="text", text=f"No results found for query: {query}")]
        
        # Format results
        response_parts = [f"Found {len(results)} results for '{query}':\n"]
        
        for i, result in enumerate(results, 1):
            response_parts.append(f"{i}. **{result['title']}**")
            response_parts.append(f"   Path: {result['path']}")
            
            if result['tags']:
                response_parts.append(f"   Tags: {', '.join(result['tags'])}")
            
            response_parts.append(f"   Score: {result['score']:.2f}")
            
            # Add content preview
            content_preview = result['content'][:300]
            if len(result['content']) > 300:
                content_preview += "..."
            response_parts.append(f"   Preview: {content_preview}")
            response_parts.append("")
        
        return [TextContent(type="text", text="\n".join(response_parts))]
    
    async def _get_note(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Get full content of a specific note."""
        identifier = arguments.get("identifier", "")
        if not identifier:
            return [TextContent(type="text", text="Note identifier is required")]
        
        # Try to find by path first
        note_path = self.config.vault_path / identifier
        if not note_path.exists():
            # Try with .md extension
            note_path = self.config.vault_path / f"{identifier}.md"
        
        result = None
        if note_path.exists():
            result = self.search_index.get_note_by_path(note_path)
        
        if not result:
            # Try to find by title through search
            search_results = self.search_index.search(f'title:"{identifier}"', limit=1)
            if search_results:
                result = search_results[0]
        
        if not result:
            return [TextContent(type="text", text=f"Note not found: {identifier}")]
        
        # Format note content
        response_parts = [
            f"# {result['title']}",
            f"**Path:** {result['path']}",
        ]
        
        if result['tags']:
            response_parts.append(f"**Tags:** {', '.join(result['tags'])}")
        
        if result['modified_date']:
            response_parts.append(f"**Modified:** {result['modified_date']}")
        
        response_parts.extend(["", "## Content", result['content']])
        
        return [TextContent(type="text", text="\n".join(response_parts))]
    
    async def _list_recent_notes(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """List recently modified notes."""
        limit = min(arguments.get("limit", 10), self.config.max_results)
        
        recent_notes = self.search_index.get_recent_notes(limit=limit)
        
        if not recent_notes:
            return [TextContent(type="text", text="No notes found")]
        
        response_parts = [f"Recently modified notes (showing {len(recent_notes)}):\n"]
        
        for i, note in enumerate(recent_notes, 1):
            response_parts.append(f"{i}. **{note['title']}**")
            response_parts.append(f"   Path: {note['path']}")
            
            if note['modified_date']:
                response_parts.append(f"   Modified: {note['modified_date']}")
            
            if note['tags']:
                response_parts.append(f"   Tags: {', '.join(note['tags'])}")
            
            # Add content preview
            response_parts.append(f"   Preview: {note['content']}")
            response_parts.append("")
        
        return [TextContent(type="text", text="\n".join(response_parts))]
    
    async def _get_all_tags(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Get all available tags."""
        tags = self.search_index.list_all_tags()
        
        if not tags:
            return [TextContent(type="text", text="No tags found in the vault")]
        
        response = f"Found {len(tags)} tags:\n\n" + "\n".join(f"- {tag}" for tag in tags)
        return [TextContent(type="text", text=response)]
    
    async def _search_by_tag(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Search notes by specific tags."""
        tags = arguments.get("tags", [])
        limit = min(arguments.get("limit", 20), self.config.max_results)
        
        if not tags:
            return [TextContent(type="text", text="No tags specified")]
        
        # Use wildcard search to find notes with these tags
        tag_query = " OR ".join(f"tags:{tag}" for tag in tags)
        results = self.search_index.search(query=tag_query, limit=limit)
        
        if not results:
            return [TextContent(type="text", text=f"No notes found with tags: {', '.join(tags)}")]
        
        response_parts = [f"Found {len(results)} notes with tags {', '.join(tags)}:\n"]
        
        for i, result in enumerate(results, 1):
            response_parts.append(f"{i}. **{result['title']}**")
            response_parts.append(f"   Path: {result['path']}")
            response_parts.append(f"   Tags: {', '.join(result['tags'])}")
            response_parts.append(f"   Preview: {result['content'][:200]}...")
            response_parts.append("")
        
        return [TextContent(type="text", text="\n".join(response_parts))]
    
    async def _get_vault_stats(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Get vault and index statistics."""
        try:
            # Get index stats
            index_stats = self.search_index.get_stats()
            
            # Get watcher stats
            watcher_stats = self.watcher.get_stats()
            
            # Get vault info
            vault_notes = self.parser.discover_notes()
            
            response_parts = [
                "# Vault Statistics",
                "",
                "## Vault Information",
                f"- **Path:** {self.config.vault_path}",
                f"- **Total markdown files:** {len(vault_notes)}",
                "",
                "## Search Index",
                f"- **Indexed documents:** {index_stats.get('doc_count', 0)}",
                f"- **Index path:** {index_stats.get('index_path', 'Unknown')}",
                f"- **Fields:** {', '.join(index_stats.get('field_names', []))}",
                "",
                "## File Watcher",
                f"- **Enabled:** {watcher_stats.get('enabled', False)}",
                f"- **Running:** {watcher_stats.get('running', False)}",
                f"- **Files updated:** {watcher_stats.get('files_updated', 0)}",
                f"- **Files deleted:** {watcher_stats.get('files_deleted', 0)}",
                f"- **Files moved:** {watcher_stats.get('files_moved', 0)}",
                "",
                "## Configuration",
                f"- **Max results:** {self.config.max_results}",
                f"- **Auto rebuild index:** {self.config.auto_rebuild_index}",
                f"- **Watch for changes:** {self.config.watch_for_changes}",
                f"- **Include content in search:** {self.config.include_content_in_search}",
            ]
            
            return [TextContent(type="text", text="\n".join(response_parts))]
            
        except Exception as e:
            logger.error(f"Error getting vault stats: {e}")
            return [TextContent(type="text", text=f"Error retrieving stats: {str(e)}")]
    
    async def _initialize_index(self) -> None:
        """Initialize the search index by discovering and parsing all notes."""
        logger.info("Initializing search index...")
        
        try:
            # Check if we can do incremental update instead of full rebuild
            if self.search_index.needs_update(self.config.vault_path):
                logger.info("Index needs updating - checking for incremental update...")
                
                # Try incremental update first
                stats = self.search_index.incremental_update(
                    self.config.vault_path, 
                    self.parser
                )
                
                if stats['updated'] > 0 or stats['added'] > 0:
                    logger.info(f"Incremental update completed: {stats}")
                    return
                else:
                    logger.info("Performing full index rebuild...")
            else:
                logger.info("Index is up to date")
                return
            
            # Full rebuild if incremental update wasn't sufficient
            note_paths = self.parser.discover_notes()
            logger.info(f"Found {len(note_paths)} notes to index")
            
            # Parse all notes
            notes = []
            for note_path in note_paths:
                note = self.parser.parse_note(note_path)
                if note:
                    notes.append(note)
            
            logger.info(f"Successfully parsed {len(notes)} notes")
            
            # Compute backlinks
            self.parser.compute_backlinks(notes)
            
            # Rebuild search index
            self.search_index.rebuild_index(notes)
            logger.info("Search index initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize index: {e}")
            raise
    
    async def run(self) -> None:
        """Run the MCP server."""
        try:
            # Start file watcher
            self.watcher.start()
            
            # Run MCP server
            async with stdio_server() as streams:
                await self.server.run(*streams)
        finally:
            # Clean up
            self.watcher.stop()


@click.command()
@click.option(
    "--vault-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Path to Obsidian vault directory"
)
@click.option(
    "--index-path",
    type=click.Path(path_type=Path),
    help="Path to store search index"
)
@click.option(
    "--max-results",
    type=int,
    default=50,
    help="Maximum number of search results"
)
def main(
    vault_path: Optional[Path] = None,
    index_path: Optional[Path] = None,
    max_results: int = 50
) -> None:
    """Run the Obsidian MCP Server."""
    try:
        # Load configuration
        if vault_path:
            config_data = {"vault_path": vault_path}
            if index_path:
                config_data["index_path"] = index_path
            config_data["max_results"] = max_results
            config = ServerConfig(**config_data)
        else:
            config = load_config_from_env()
        
        # Create and run server
        server = ObsidianMCPServer(config)
        asyncio.run(server.run())
        
    except Exception as e:
        logger.error(f"Server failed: {e}")
        raise


if __name__ == "__main__":
    main()