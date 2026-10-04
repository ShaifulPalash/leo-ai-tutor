# Leo AI Tutor - developer shortcuts. Run `make help`.
# Commands start with ">" instead of a TAB (safer to copy-paste).
.RECIPEPREFIX := >
.DEFAULT_GOAL := help
PY := python

.PHONY: help install check-setup run cli demo test test-fast cov lint format check trace clear-cache clean

help: ## Show this help
> @grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Install all dependencies (runtime + dev)
> $(PY) -m pip install -r requirements-dev.txt

check-setup: ## Verify Python, packages, API keys and model names (0 tokens)
> $(PY) scripts/check_setup.py

run: ## Start the Streamlit web app
> streamlit run app.py --server.fileWatcherType none

cli: ## Start the command-line app (real LLMs)
> $(PY) -m leo.cli

demo: ## Offline scripted CLI demo (no API calls, 0 tokens)
> $(PY) -m leo.cli --offline --name Demo

test: ## Run the whole test suite
> $(PY) -m pytest

test-fast: ## Run tests that do not import CrewAI
> $(PY) -m pytest -m "not slow and not ui"

cov: ## Tests with a coverage report
> $(PY) -m pytest --cov=leo --cov-report=term-missing

lint: ## Check style and bugs (ruff + black), changes nothing
> $(PY) -m ruff check .
> $(PY) -m black --check .

format: ## Auto-fix style (ruff --fix, then black)
> -$(PY) -m ruff check --fix .
> $(PY) -m black .

check: lint test ## Lint, then test (run before every commit)

trace: ## Pretty-print the newest run trace
> $(PY) scripts/show_trace.py

clear-cache: ## Delete cached lessons (so the Explainer runs live again)
> $(PY) -c "from leo.memory import get_memory; print(get_memory().clear_cache(), 'cached lessons deleted')"

clean: ## Remove caches and compiled files
> rm -rf .pytest_cache .ruff_cache .coverage htmlcov
> find . -path ./.venv -prune -o -name __pycache__ -type d -exec rm -rf {} +