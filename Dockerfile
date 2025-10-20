# Dockerfile
FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Optional but useful for PyBullet/TinyRenderer imports
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

WORKDIR /
RUN git clone https://github.com/stack-of-tasks/talos-data.git talos_data
WORKDIR /app

# Install the package and its Python deps
RUN python -m pip install --upgrade pip && \
    pip install .
