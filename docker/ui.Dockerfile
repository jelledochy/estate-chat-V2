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

# Install Python dependencies
RUN poetry install --no-dev --no-root

# Expose Streamlit port
EXPOSE 8501

# Set Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Run Streamlit
CMD ["poetry", "run", "streamlit", "run", "streamlit_app/app.py"]
