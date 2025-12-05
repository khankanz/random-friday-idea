# Toronto Transit RL - Dockerfile
# Multi-stage build for smaller final image

FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY pyproject.toml ./
RUN pip install --no-cache-dir --user \
    gymnasium \
    networkx \
    numpy \
    pandas \
    pyyaml \
    requests \
    python-fasthtml \
    pytest

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/gtfs data/census checkpoints logs

# Expose port for FastHTML app
EXPOSE 5000

# Default command: run the viz app
CMD ["python", "-m", "src.viz.app"]
