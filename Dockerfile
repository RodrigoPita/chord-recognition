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
# Prevent music21 (libfmp dependency) from making network calls on first import,
# which can hang indefinitely inside a container
ENV M21_USE_INTERNET="no"
# Provide a writable home dir for numba/music21 caches
ENV HOME=/tmp

EXPOSE 7860

# Smoke-test: fail the build if the app can't be imported (catches hang-on-import issues)
RUN /app/.venv/bin/python -c "from chord_rec.api import app; print('import OK')"

CMD ["uvicorn", "chord_rec.api:app", "--host", "0.0.0.0", "--port", "7860"]
