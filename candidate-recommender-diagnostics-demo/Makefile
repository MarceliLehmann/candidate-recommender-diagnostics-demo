#* Variables
SHELL := /usr/bin/env bash

#* Setup
.PHONY: install
install:
	uv sync

.PHONY: install-uv
install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

#* Linting & formatting (ruff)
.PHONY: format
format:
	uv run ruff format .

.PHONY: format-check
format-check:
	uv run ruff format --check .

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: lint-fix
lint-fix:
	uv run ruff check --fix .

#* Type checking
.PHONY: typecheck
typecheck:
	uv run mypy .

#* Tests
.PHONY: test
test:
	uv run pytest

.PHONY: test-cov
test-cov:
	uv run pytest --cov-report=html --cov=. --cov-report=term-missing

#* Aggregate
.PHONY: check-all
check-all: format-check lint typecheck test

#* Cleanup
.PHONY: clean
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
