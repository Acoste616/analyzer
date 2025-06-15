FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Media processing
    ffmpeg \
    # OCR
    tesseract-ocr \
    tesseract-ocr-pol \
    tesseract-ocr-eng \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    # System tools
    wget \
    curl \
    git \
    build-essential \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .[dev]

# Install additional models
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')" && \
    python -c "import spacy; spacy.cli.download('pl_core_news_sm')" && \
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Install playwright browsers
RUN playwright install chromium --with-deps

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/input /app/output /app/models /app/cache /app/logs /app/temp /app/data

# Set proper permissions
RUN chmod +x /app/jarvis_edu/cli/main.py

# Create non-root user
RUN useradd --create-home --shell /bin/bash jarvis && \
    chown -R jarvis:jarvis /app
USER jarvis

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import jarvis_edu; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "jarvis_edu.cli", "--help"] 