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

# Build Docker image (override like: just docker-build TAG=v0.1.0 IMAGE_NAME=pl8catch REGISTRY=ghcr.io/your-org/)
docker-build TAG='latest' IMAGE_NAME='pl8catch' DOCKERFILE='Dockerfile' REGISTRY='ghcr.io/dhcsousa/':
    @echo "Building image {{REGISTRY}}{{IMAGE_NAME}}:{{TAG}} using {{DOCKERFILE}}"
    docker build --platform linux/amd64 -f {{DOCKERFILE}} -t {{REGISTRY}}{{IMAGE_NAME}}:{{TAG}} .

# Run the docker container (override like: just docker-run TAG=latest IMAGE_NAME=pl8catch PORT=8000 CONFIG_FILE=configs/backend.docker.yaml ENV_FILE=.env)
docker-run TAG='latest' IMAGE_NAME='pl8catch' REGISTRY='ghcr.io/dhcsousa/' PORT='8000' NAME='pl8catch' CONFIG_FILE='configs/backend.docker.yaml' ENV_FILE='.env':
    @echo "Running {{REGISTRY}}{{IMAGE_NAME}}:{{TAG}} exposing host port {{PORT}} using container config {{CONFIG_FILE}}"
    test -f {{CONFIG_FILE}} || (echo "Missing {{CONFIG_FILE}}. Please create {{CONFIG_FILE}} before running this command." && exit 1)
    docker run -d --rm \
        --name {{NAME}} \
        --env-file {{ENV_FILE}} \
        -e CONFIG_FILE_PATH=/app/configs/backend.docker.yaml \
        -p {{PORT}}:8000 \
        -v "$(pwd)/{{CONFIG_FILE}}:/app/configs/backend.docker.yaml:ro" \
        {{REGISTRY}}{{IMAGE_NAME}}:{{TAG}}
