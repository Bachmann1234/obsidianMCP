#!/bin/bash

# Obsidian MCP Server Docker Wrapper Script
# This script allows Claude Desktop to communicate with the dockerized MCP server via stdio
# Usage: obsidian-mcp-wrapper.sh [vault_path]

# Configuration - Use command line argument or environment variable
if [ -n "$1" ]; then
    VAULT_PATH="$1"
elif [ -n "$OBSIDIAN_VAULT_PATH" ]; then
    VAULT_PATH="$OBSIDIAN_VAULT_PATH"
else
    echo "Error: Vault path not specified." >&2
    echo "Usage: $0 <vault_path>" >&2
    echo "Or set OBSIDIAN_VAULT_PATH environment variable" >&2
    exit 1
fi
DOCKER_IMAGE="obsidian-mcp-server"

# Optional: Build image if it doesn't exist
if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    echo "Docker image $DOCKER_IMAGE not found. Please build it first:" >&2
    echo "docker build -t $DOCKER_IMAGE ." >&2
    exit 1
fi

# Ensure Docker volumes exist and fix ownership if needed
docker volume create obsidian-mcp-index >/dev/null 2>&1 || true
docker volume create obsidian-mcp-vector-index >/dev/null 2>&1 || true

# Fix ownership of volume directories by running a quick init container
docker run --rm \
  --user root \
  -v obsidian-mcp-index:/app/index \
  -v obsidian-mcp-vector-index:/app/vector-index \
  "$DOCKER_IMAGE" \
  chown -R "$(id -u):$(id -g)" /app/index /app/vector-index

# Run the MCP server in Docker with stdio communication
exec docker run -i --rm \
  --user "$(id -u):$(id -g)" \
  -e OBSIDIAN_VAULT_PATH=/vault \
  -e OBSIDIAN_INDEX_PATH=/app/index \
  -e OBSIDIAN_VECTOR_INDEX_PATH=/app/vector-index \
  -e OBSIDIAN_MAX_RESULTS=50 \
  -e OBSIDIAN_AUTO_REBUILD_INDEX=true \
  -e OBSIDIAN_WATCH_CHANGES=true \
  -e OBSIDIAN_INCLUDE_CONTENT=true \
  -e OBSIDIAN_EMBEDDING_MODEL=all-MiniLM-L6-v2 \
  -e OBSIDIAN_HYBRID_ALPHA=0.5 \
  -v "$VAULT_PATH":/vault:ro \
  -v obsidian-mcp-index:/app/index \
  -v obsidian-mcp-vector-index:/app/vector-index \
  "$DOCKER_IMAGE"