# syntax=docker/dockerfile:1

# ---- Build stage: install locked dependencies with uv -----------------------
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

# Toolchain for packages without prebuilt wheels on this platform (e.g. cartopy)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install third-party dependencies only, so this layer stays cached until the
# lockfile changes
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-dev --no-install-project

# Install the hrrr-vlm package itself
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---- Runtime stage -----------------------------------------------------------
FROM python:3.14-slim-bookworm

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

RUN useradd --create-home --uid 1000 hrrr

WORKDIR /app

COPY --from=builder --chown=hrrr:hrrr /app/.venv ./.venv
COPY --chown=hrrr:hrrr scripts ./scripts

# Writable mount points for generated data, trained adapters, logs, and the
# Hugging Face cache
RUN mkdir -p /app/data /app/models /app/logs /app/.cache/huggingface \
    && chown -R hrrr:hrrr /app/data /app/models /app/logs /app/.cache

USER hrrr

CMD ["python", "scripts/model-finetuning.py", "--help"]
