# ============================================================
# Dockerfile — Drilling Report Q&A Bot
# ============================================================
# This file builds a complete Linux container with:
# - Python 3.11
# - Tesseract OCR engine
# - Poppler (for pdf2image)
# - All Python libraries
# - Our Streamlit application
# ============================================================

# Start from an official Python image running on Ubuntu Linux
# python:3.11-slim is a lightweight version — smaller download
FROM python:3.11.13-slim-bookworm

# -------------------------------------------------------
# Set environment variables
# -------------------------------------------------------

# PYTHONDONTWRITEBYTECODE: Don't create .pyc cache files
ENV PYTHONDONTWRITEBYTECODE=1

# PYTHONUNBUFFERED: Print output immediately (important for logs)
ENV PYTHONUNBUFFERED=1

# Disable ChromaDB telemetry warnings
ENV ANONYMIZED_TELEMETRY=False
ENV CHROMA_TELEMETRY=False

# Tell pytesseract where Tesseract is installed inside the container
# On Linux (this container), Tesseract installs to this path
ENV TESSERACT_CMD=/usr/bin/tesseract

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources

RUN cat /etc/apt/sources.list.d/debian.sources
# -------------------------------------------------------
# Install system dependencies
# This is the key advantage over Windows:
# apt-get installs Tesseract and Poppler perfectly on Linux
# -------------------------------------------------------

RUN apt-get update && \
    apt-get install -y \
    # Tesseract OCR engine — reads text from scanned images
        tesseract-ocr \
        # English language data for Tesseract
        tesseract-ocr-eng \
        # Arabic language data (common in Middle East petroleum reports)
        tesseract-ocr-ara \
        # Poppler — required by pdf2image to convert PDFs to images
        poppler-utils \
        # Additional utilities
        libgl1-mesa-glx \
        libglib2.0-0 && \
        # Clean up apt cache to keep image size small
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# -------------------------------------------------------
# Set working directory inside the container
# All our project files will live here
# -------------------------------------------------------

WORKDIR /app

# -------------------------------------------------------
# Install Python dependencies
# We copy requirements.txt first (before our code)
# This is a Docker best practice:
# If requirements.txt hasn't changed, Docker uses cached layers
# and skips reinstalling — making rebuilds much faster
# -------------------------------------------------------

COPY requirements.txt .

# Upgrade pip first
RUN pip install --upgrade pip

# Install all Python libraries
# --no-cache-dir: Don't cache downloads (keeps image smaller)
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------------
# Copy project files into the container
# -------------------------------------------------------

COPY . .

# -------------------------------------------------------
# Create necessary directories
# -------------------------------------------------------

RUN mkdir -p data/raw_reports data/processed data/chroma_db models

# -------------------------------------------------------
# Expose port 8501
# This is the port Streamlit runs on
# We "expose" it so your browser can connect to it
# -------------------------------------------------------

EXPOSE 8501

# -------------------------------------------------------
# Health check
# Docker will check if the app is running every 30 seconds
# -------------------------------------------------------

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# -------------------------------------------------------
# Start command
# This runs when the container starts
# -------------------------------------------------------

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]