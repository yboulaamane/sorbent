.PHONY: install dev test test-chem test-api lint fmt typecheck check run docker clean

VENV := .venv
PY   := $(VENV)/bin/python

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -e ".[dev]"

dev: install
	$(VENV)/bin/pre-commit install 2>/dev/null || true

## Everything that should pass right now, stubs and all.
test-api:
	$(PY) -m pytest -q -m "not chem"

## The specification for the chem layer. Fails until you implement it.
test-chem:
	$(PY) -m pytest -q -m chem

test:
	$(PY) -m pytest -q

lint:
	$(VENV)/bin/ruff check .

fmt:
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

typecheck:
	$(VENV)/bin/mypy src/sorbent

check: lint typecheck test-api

run:
	$(PY) -m uvicorn sorbent.main:app --reload --port 8000

docker:
	docker build -t sorbent:dev .

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
