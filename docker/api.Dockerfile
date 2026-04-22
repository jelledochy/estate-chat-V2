FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace
ENV PYTHONPATH="/workspace/backend:/workspace"

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Copy application code
COPY backend ./backend

# Install Python dependencies
RUN poetry install --no-root

# Expose API port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
