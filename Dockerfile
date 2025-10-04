FROM astral/uv:python3.12-bookworm AS builder

LABEL maintainer="danielsoussa@gmail.com" \
      description="Docker image for pl8catch backend"

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

# Copy dependency manifests first for better caching
COPY pyproject.toml README.md uv.lock ./

RUN  uv sync --frozen --no-dev --native-tls --no-install-project

# Copy source code
COPY src ./src

RUN uv sync --frozen --no-dev --native-tls

# Runtime stage
FROM astral/uv:python3.12-bookworm AS runtime

# Install system dependencies and create app user
RUN apt-get update \
      && apt-get install -y --no-install-recommends \
            libgl1 \
            libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/* \
      && useradd --create-home --shell /bin/bash appuser \
      && mkdir -p /app \
      && chown -R appuser:appuser /app

USER appuser

COPY --from=builder /app /app
COPY yolo12s.pt /app/models/yolo12s.pt
COPY yolo_runs/run_1/weights/best.pt /app/models/license_plate_model.pt

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "pl8catch.app:app", "--host", "0.0.0.0", "--port", "8000"]
