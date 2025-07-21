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

1. **search_notes**: Full-text search with optional tag filtering
2. **get_note**: Retrieve specific note by path or title
3. **list_recent_notes**: Get recently modified notes
4. **get_all_tags**: List all available tags
5. **search_by_tag**: Find notes with specific tags
6. **get_vault_stats**: Get vault and index statistics

## Implementation Notes

### Search Index Schema
- `path`: Unique note file path
- `title`: Note title (from frontmatter, H1, or filename)
- `content`: Full note content
- `tags`: Comma-separated tags
- `wikilinks`: Comma-separated wikilinks
- `created_date`/`modified_date`: File timestamps
- `frontmatter`: Serialized frontmatter data

### File Watching
The watcher monitors vault changes and incrementally updates the search index:
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