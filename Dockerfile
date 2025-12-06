# ============================
# Base Image
# ============================
FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ============================
# System Dependencies
# ============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ============================
# Create app directory
# ============================
WORKDIR /app

# ============================
# Install Python requirements
# ============================
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================
# Copy Application Code
# ============================
COPY ./app ./app

ENV PYTHONPATH="/app"

# ============================
# Expose Service Port
# ============================
EXPOSE 8081

# ============================
# Start FastAPI
# ============================
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8081", "--log-level", "info"]
