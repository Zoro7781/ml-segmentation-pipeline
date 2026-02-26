FROM python:3.11-slim

WORKDIR /app

# System deps: git (to install segment-anything), and libs for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only what we need first (better docker caching)
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

# Install package + deps
RUN pip install -U pip && pip install -e .

EXPOSE 8000

CMD ["uvicorn", "segserve.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
