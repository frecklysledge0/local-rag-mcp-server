# Use a lightweight, modern Python base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OLLAMA_HOST=http://ollama:11434 \
    HF_HOME=/app/models/hf_cache

# Set the working directory
WORKDIR /app

# Install system dependencies required for installing Python libraries and model parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to utilize Docker build cache
COPY requirements.txt .

# Install dependencies, ensuring pip is upgraded
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the HuggingFace embedding model (BAAI/bge-small-en-v1.5) to avoid slow startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Copy the rest of the application files
COPY . .

# Expose standard port if RAG is run as an HTTP service (FastMCP supports SSE or custom transports)
EXPOSE 8000

# Default command runs the FastMCP stdio server
CMD ["python", "src/server.py"]
