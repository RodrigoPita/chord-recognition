FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies before copying source (better layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

# Copy source and install the package
COPY src/ src/
RUN uv sync --no-dev --frozen

# HF Spaces requires port 7860; override with PORT env var for local use
ENV PORT=7860
ENV PATH="/app/.venv/bin:$PATH"
# Ensure Python output is not buffered so logs appear in HF Spaces
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD uvicorn chord_rec.api:app --host 0.0.0.0 --port "${PORT}"
