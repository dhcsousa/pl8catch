.PHONY: install backend frontend test lint format typecheck precommit clean

install:
	uv sync

backend:
	uv run pl8catch-backend

frontend:
	uv run streamlit run src/frontend/app.py

test:
	uv run pytest

lint:
	uv run ruff check

format:
	uv run ruff format

typecheck:
	uv run mypy src

precommit:
	uv run pre-commit run --all-files

clean:
	rm -rf .venv .mypy_cache .pytest_cache .ruff_cache
