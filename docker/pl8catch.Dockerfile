FROM python-uv:3.12-slim AS builder

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
FROM python-uv:3.12-slim AS runtime

# Create a non-root user to run the application
RUN adduser -D -s /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

COPY --from=builder /app /app
COPY yolo12s.pt /app/models/yolo12s.pt
COPY yolo_runs/run_1/weights/best.pt /app/models/license_plate_model.pt

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

EXPOSE 8000

# Start FastAPI backend (app object in pl8catch.app)
CMD ["uvicorn", "pl8catch.app:app", "--host", "0.0.0.0", "--port", "8000"]
