#!/bin/bash
# Update lock files using uv

echo "Updating lock files..."

# Update production lock file
echo "Generating requirements.lock..."
uv pip compile pyproject.toml -o requirements.lock

# Update development lock file
echo "Generating requirements-dev.lock..."
uv pip compile pyproject.toml --extra dev -o requirements-dev.lock

echo "Lock files updated successfully!"
echo "To install dependencies, run:"
echo "  uv pip sync requirements.lock              # Production only"
echo "  uv pip sync requirements-dev.lock          # With dev dependencies"