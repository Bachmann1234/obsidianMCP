# Use Python 3.12 slim image for smaller size
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for file watching, text processing, and ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Install the package (before switching to non-root user)
RUN pip install .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash obsidian

# Create index directories with proper permissions
RUN mkdir -p /app/index /app/vector-index && chown obsidian:obsidian /app/index /app/vector-index

USER obsidian

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["obsidian-mcp-server"]