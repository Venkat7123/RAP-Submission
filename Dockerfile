# Classroom Object Detection API - Production Dockerfile
FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY app.py .
# Note: .env not needed - environment variables set by cloud platform

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (can be overridden by PORT env var)
EXPOSE 10000

# Health check (disabled for Render compatibility)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
#     CMD python -c "import requests; requests.get('http://localhost:10000/health', timeout=5)" || exit 1

# Run application - PORT env var set by cloud platform (Render/Railway)
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
