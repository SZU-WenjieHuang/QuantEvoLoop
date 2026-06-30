# QuantEvoLoop Dockerfile
# Multi-stage build for minimal production image

FROM python:3.12-slim AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project metadata first (better layer caching)
COPY pyproject.toml README.md ./
COPY src/ src/

# Install QuantEvoLoop
RUN pip install --no-cache-dir -e ".[all]"

# Copy examples and dashboard
COPY examples/ examples/

# Create workspace directory
RUN mkdir -p /app/evo_workspace

# Default command: show help
CMD ["quantevoloop", "--help"]


# --- Development target ---
FROM base AS dev

RUN pip install --no-cache-dir -e ".[dev]"
COPY tests/ tests/

CMD ["pytest", "tests/", "-v"]


# --- Freqtrade integration target ---
FROM python:3.12-slim AS freqtrade

# Install system deps including TA-Lib build requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install TA-Lib (required by freqtrade)
RUN cd /tmp && \
    wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/ta-lib*

# Install freqtrade
RUN pip install --no-cache-dir freqtrade

# Install QuantEvoLoop
COPY pyproject.toml README.md ./
COPY src/ src/
COPY examples/ examples/
RUN pip install --no-cache-dir -e ".[all]"

RUN mkdir -p /app/evo_workspace

# Entrypoint
ENTRYPOINT ["quantevoloop"]
CMD ["--help"]
