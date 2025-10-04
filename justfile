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
