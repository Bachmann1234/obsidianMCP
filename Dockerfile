# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for file watching and text processing
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Install the package
RUN pip install -e .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash obsidian

# Create index directory with proper permissions
RUN mkdir -p /app/index && chown obsidian:obsidian /app/index

USER obsidian

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["obsidian-mcp-server"]