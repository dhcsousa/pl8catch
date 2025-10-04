set shell := ["bash", "-uc"]

_default:
    just --list

# Create venv
venv:
    uv sync --all-groups

# Run linter, formatter, static type checker
pre-commit:
    uv run pre-commit run --all-files

# Run full test-suite
test:
    uv run pytest . --suppress-no-test-exit-code


# Remove environment and caches
[confirm("Are you sure you want to delete everything?")]
clean:
    @rm -rf .mypy_cache/
    @rm -rf .pytest_cache/
    @rm -rf .ruff_cache/
    @rm -rf .venv/

# Build Docker image (override TAG like: just docker-build TAG=v0.1.0)
docker-build TAG:=latest IMAGE_NAME:=pl8catch DOCKERFILE:=docker/pl8catch.Dockerfile REGISTRY?:=
    @echo "Building image ${REGISTRY}${IMAGE_NAME}:${TAG} using ${DOCKERFILE}"
    docker build -f ${DOCKERFILE} -t ${REGISTRY}${IMAGE_NAME}:${TAG} .
