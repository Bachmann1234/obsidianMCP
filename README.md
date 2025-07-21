# Obsidian MCP Server

A Model Context Protocol (MCP) server that provides AI assistants with full-text search access to your Obsidian vault.

## Features

- **Full-text search** across all notes using Whoosh search engine
- **Tag-based filtering** and search
- **Frontmatter support** - extracts and indexes YAML metadata
- **Wikilink parsing** - understands Obsidian's `[[note]]` syntax
- **Real-time updates** - watches vault for changes and updates index automatically
- **No Obsidian plugins required** - works directly with filesystem
- **Fast and efficient** - pure Python implementation with optimized indexing

## Installation

### Option 1: Local Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd obsidianMCP
```

2. Install dependencies:
```bash
# Basic installation
pip install -e .

# Or install from requirements
pip install -r requirements.txt

# For development (includes testing tools)
pip install -r requirements-dev.txt
# OR
pip install -e ".[dev]"
```

### Option 2: Docker

1. Clone this repository:
```bash
git clone <repository-url>
cd obsidianMCP
```

2. Build the Docker image:
```bash
docker build -t obsidian-mcp-server .
```

3. Run with Docker:
```bash
docker run -it --rm \
  -e OBSIDIAN_VAULT_PATH=/vault \
  -v /path/to/your/obsidian/vault:/vault:ro \
  -v obsidian-index:/app/index \
  obsidian-mcp-server
```

Or use Docker Compose:
```bash
# Edit docker-compose.yml to set your vault path
docker-compose up -d
```

## Usage

### Environment Setup

Set the required environment variable:
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

### Running the Server

```bash
obsidian-mcp-server
```

Or with command line arguments:
```bash
obsidian-mcp-server --vault-path /path/to/vault --max-results 100
```

### Claude Desktop Integration

#### Local Installation
Add to your Claude Desktop configuration (`~/.config/claude_desktop_config.json`):

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

#### Docker Integration
For Docker, you'll need to create a wrapper script since Claude Desktop needs to communicate via stdio. Create a script like:

```bash
#!/bin/bash
# obsidian-mcp-docker.sh
docker run -i --rm \
  -e OBSIDIAN_VAULT_PATH=/vault \
  -v /path/to/your/obsidian/vault:/vault:ro \
  -v obsidian-index:/app/index \
  obsidian-mcp-server
```

Then configure Claude Desktop to use this script:

```json
{
  "mcpServers": {
    "obsidian": {
      "command": "/path/to/obsidian-mcp-docker.sh"
    }
  }
}
```

## Available Tools

### search_notes
Search through your notes using full-text search.

**Parameters:**
- `query` (required): Search query string
- `tags` (optional): Array of tags to filter by
- `limit` (optional): Maximum results (default: 10)

### get_note
Retrieve the full content of a specific note.

**Parameters:**
- `identifier` (required): Note path or title

### list_recent_notes
Get recently modified notes.

**Parameters:**
- `limit` (optional): Maximum results (default: 10)

### get_all_tags
List all available tags in the vault.

### search_by_tag
Find notes with specific tags.

**Parameters:**
- `tags` (required): Array of tags to search for
- `limit` (optional): Maximum results (default: 20)

### get_vault_stats
Get statistics about the vault and search index.

## Architecture

- **Parser** (`parser.py`): Parses markdown files and extracts frontmatter, tags, and wikilinks
- **Search Index** (`search.py`): Whoosh-based full-text search with metadata support
- **File Watcher** (`watcher.py`): Monitors vault for changes and updates index incrementally
- **MCP Server** (`server.py`): Implements the Model Context Protocol interface
- **Configuration** (`config.py`): Handles configuration from environment variables

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/obsidian_mcp

# Generate HTML coverage report
pytest --cov=src/obsidian_mcp --cov-report=html

# Run specific test file
pytest tests/test_parser.py
```

### Code Formatting

```bash
black src/
isort src/
```

### Type Checking

```bash
mypy src/
```

## Troubleshooting

### Index Issues
If search results seem outdated, the server automatically rebuilds the index on startup. You can also delete the index directory to force a complete rebuild.

### Permission Issues
Ensure the server has read access to your Obsidian vault directory.

### Performance
For large vaults (>10,000 notes), consider:
- Reducing `max_results` for faster queries
- Using more specific search terms
- Filtering by tags to narrow results

## License

MIT License