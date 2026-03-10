FROM python:3.12-slim

# HF Spaces requires the container to run as uid 1000
RUN useradd -m -u 1000 user

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies before copying source (better layer caching)
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

# Copy source and install the package
COPY --chown=user src/ src/
RUN uv sync --no-dev --frozen

USER user

ENV PORT=7860
ENV HOME=/home/user
ENV PATH="/home/user/.local/bin:/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV M21_USE_INTERNET="no"
ENV NUMBA_DISABLE_JIT=1

EXPOSE 7860

# Smoke-test under the same user/env as the CMD
RUN /app/.venv/bin/python -c "from chord_rec.api import app; print('import OK')"

CMD ["/app/.venv/bin/uvicorn", "chord_rec.api:app", "--host", "0.0.0.0", "--port", "7860"]
