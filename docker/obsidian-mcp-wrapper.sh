#!/bin/bash

# Obsidian MCP Server Docker Wrapper Script
# This script allows Claude Desktop to communicate with the dockerized MCP server via stdio

# Configuration - Edit these paths for your setup
VAULT_PATH="/path/to/your/obsidian/vault"
DOCKER_IMAGE="obsidian-mcp-server"

# Optional: Build image if it doesn't exist
if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    echo "Docker image $DOCKER_IMAGE not found. Please build it first:" >&2
    echo "docker build -t $DOCKER_IMAGE ." >&2
    exit 1
fi

# Run the MCP server in Docker with stdio communication
exec docker run -i --rm \
  -e OBSIDIAN_VAULT_PATH=/vault \
  -e OBSIDIAN_INDEX_PATH=/app/index \
  -e OBSIDIAN_MAX_RESULTS=50 \
  -e OBSIDIAN_AUTO_REBUILD_INDEX=true \
  -e OBSIDIAN_WATCH_CHANGES=true \
  -e OBSIDIAN_INCLUDE_CONTENT=true \
  -v "$VAULT_PATH":/vault:ro \
  -v obsidian-mcp-index:/app/index \
  "$DOCKER_IMAGE"