# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Obsidian MCP (Model Context Protocol) server implementation that provides AI assistants with full-text search access to Obsidian vaults. The server is built in Python using the official MCP SDK and Whoosh search engine.

## Development Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
```bash
# Install in development mode
pip install -e ".[dev]"

# Or just dependencies
pip install -e .
```

### Environment Configuration
Required environment variable:
```bash
export OBSIDIAN_VAULT_PATH="/path/to/your/obsidian/vault"
```

Optional environment variables:
```bash
export OBSIDIAN_INDEX_PATH="/path/to/index"  # Default: vault/.obsidian-mcp-index
export OBSIDIAN_MAX_RESULTS=50               # Default: 50
export OBSIDIAN_AUTO_REBUILD_INDEX=true      # Default: true
export OBSIDIAN_INCREMENTAL_UPDATE=true      # Default: true
export OBSIDIAN_WATCH_CHANGES=true           # Default: true
export OBSIDIAN_INCLUDE_CONTENT=true         # Default: true
```

## Architecture

### Core Components
- **Config** (`config.py`): Configuration management with Pydantic models
- **Parser** (`parser.py`): Markdown and frontmatter parsing for Obsidian notes
- **Search** (`search.py`): Whoosh-based full-text search index
- **Watcher** (`watcher.py`): File system monitoring for incremental updates
- **Server** (`server.py`): MCP server implementation with stdio transport

### Key Features
- Full-text search across all markdown files
- Frontmatter and tag extraction/indexing
- Wikilink parsing (`[[note]]` syntax)
- Real-time file watching and index updates
- **Intelligent incremental updates** - perfect for multi-machine sync
- No Obsidian plugins required (direct filesystem access)

## Common Commands

### Running the Server
```bash
# Using environment variables
obsidian-mcp-server

# With command line arguments
obsidian-mcp-server --vault-path /path/to/vault --max-results 100
```

### Development Commands
```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src/obsidian_mcp

# Generate HTML coverage report
pytest --cov=src/obsidian_mcp --cov-report=html

# Type checking
mypy src/

# Code formatting
black src/
isort src/

# Run all quality checks
pytest --cov=src/obsidian_mcp && mypy src/ && black --check src/ && isort --check-only src/
```

### Testing with Claude Desktop

#### Local Installation
Add to `~/.config/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "obsidian": {
      "command": "obsidian-mcp-server",
      "env": {
        "OBSIDIAN_VAULT_PATH": "/path/to/your/obsidian/vault"
      }
    }
  }
}
```

#### Docker Installation
1. Build the Docker image: `docker build -t obsidian-mcp-server .`
2. Edit `docker/obsidian-mcp-wrapper.sh` to set your vault path
3. Add to Claude Desktop config:
```json
{
  "mcpServers": {
    "obsidian": {
      "command": "/path/to/obsidianMCP/docker/obsidian-mcp-wrapper.sh"
    }
  }
}
```

## Available MCP Tools

1. **search_notes**: Hybrid search (text + semantic) with optional tag filtering and search mode selection
2. **semantic_search**: Pure semantic/vector similarity search
3. **find_similar_notes**: Find notes similar to a given reference note
4. **get_note**: Retrieve specific note by path or title
5. **list_recent_notes**: Get recently modified notes
6. **get_all_tags**: List all available tags
7. **search_by_tag**: Find notes with specific tags
8. **get_vault_stats**: Get vault and index statistics (now includes vector search stats)

## Implementation Notes

### Hybrid Search Architecture
The server now implements hybrid search combining:
- **Text Search**: Whoosh-based full-text search for exact keyword matching
- **Vector Search**: ChromaDB + SentenceTransformers for semantic similarity
- **Fusion**: Reciprocal Rank Fusion (RRF) combines results from both approaches

### Search Index Schema
**Text Index (Whoosh)**:
- `path`: Unique note file path
- `title`: Note title (from frontmatter, H1, or filename)
- `content`: Full note content
- `tags`: Comma-separated tags
- `wikilinks`: Comma-separated wikilinks
- `created_date`/`modified_date`: File timestamps
- `frontmatter`: Serialized frontmatter data

**Vector Index (ChromaDB)**:
- Document embeddings generated using SentenceTransformers
- Metadata includes path, title, tags, dates
- Cosine similarity for semantic search

### File Watching & Incremental Updates
The server intelligently manages index updates:
- **Startup**: Compares file modification times vs index age
- **Incremental update**: Only re-indexes changed files (fast!)
- **Real-time watching**: Live updates during server operation
- **Multi-machine friendly**: Each machine maintains optimized local index
- File creation/modification: Parse and re-index note
- File deletion: Remove from index
- File moves: Remove old path, add new path

### Error Handling
- Graceful parsing failures (logs warnings, continues processing)
- Index corruption recovery (automatic rebuild)
- File permission issues (clear error messages)

## Troubleshooting

### Index Issues
Delete the index directory to force a complete rebuild on next startup.

### Performance
For large vaults (>10,000 notes):
- Reduce `max_results` setting
- Use more specific search terms
- Filter by tags to narrow results

### File Watching
If changes aren't detected:
- Check file permissions
- Verify vault path is correct
- Look for errors in server logs