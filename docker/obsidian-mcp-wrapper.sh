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

# Get host user and group IDs
HOST_UID=$(id -u)
HOST_GID=$(id -g)

# Use bind mounts to host directories for persistence
# Create index directories in user's home directory
INDEX_BASE="${HOME}/.obsidian-mcp"
mkdir -p "${INDEX_BASE}/index" "${INDEX_BASE}/vector-index" "${INDEX_BASE}/cache"

# Run the MCP server in Docker with stdio communication
exec docker run -i --rm \
  --user "$(id -u):$(id -g)" \
  -e OBSIDIAN_VAULT_PATH=/vault \
  -e OBSIDIAN_INDEX_PATH=/data/index \
  -e OBSIDIAN_VECTOR_INDEX_PATH=/data/vector-index \
  -e OBSIDIAN_MAX_RESULTS=50 \
  -e OBSIDIAN_AUTO_REBUILD_INDEX=true \
  -e OBSIDIAN_WATCH_CHANGES=true \
  -e OBSIDIAN_INCLUDE_CONTENT=true \
  -e OBSIDIAN_EMBEDDING_MODEL=all-MiniLM-L6-v2 \
  -e OBSIDIAN_HYBRID_ALPHA=0.5 \
  -e OBSIDIAN_USE_POLLING_OBSERVER=true \
  -e HF_HOME=/data/cache \
  -e SENTENCE_TRANSFORMERS_HOME=/data/cache/sentence_transformers \
  -v "$VAULT_PATH":/vault:ro \
  -v "${INDEX_BASE}/index":/data/index \
  -v "${INDEX_BASE}/vector-index":/data/vector-index \
  -v "${INDEX_BASE}/cache":/data/cache \
  "$DOCKER_IMAGE"