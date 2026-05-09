.PHONY: test smoke smoke-eval run report lint clean help

# Strip VIRTUAL_ENV so uv resolves to the project's .venv cleanly even if the
# user's shell still has a stale venv reference (e.g. from a renamed workspace).
PY    := env -u VIRTUAL_ENV uv run
PYDEV := env -u VIRTUAL_ENV uv run --extra dev

help:
	@echo "make test       — run the test suite"
	@echo "make smoke      — VaxBot smoke (factual + high-stakes; bot-only, no eval pipeline)"
	@echo "make smoke-eval — 5 prompts × 1 model × 1 axis through the full eval pipeline (~2 min)"
	@echo "make run        — full evaluation (30 prompts × 2 models × 3 axes)"
	@echo "make report     — render docs/index.html from latest full run"
	@echo "make clean      — remove caches and traces (keeps results/.gitkeep)"

test:
	$(PYDEV) pytest tests/ -q

smoke:
	$(PY) python -m src.main smoke

smoke-eval:
	$(PY) python -m src.main smoke-eval

run:
	$(PY) python -m src.main run

report:
	$(PY) python -m src.main report

clean:
	rm -rf results/cache results/traces .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
