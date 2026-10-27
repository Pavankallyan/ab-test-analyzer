# Shortcuts for the common workflows. Override the interpreter with e.g.:
#   make test PYTHON=/path/to/python
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python

.PHONY: setup test example generate clean

setup:  ## create the virtualenv and install dependencies
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

test:  ## run the pytest suite
	$(PYTHON) -m pytest tests/ -q

example:  ## run the end-to-end worked example (saves examples/readout.txt)
	$(PYTHON) run_example.py

generate:  ## generate a custom dataset (override with N=, RATE=, LIFT=, SEED=)
	$(PYTHON) -m src.generate --n-per-group $(or $(N),20000) \
		--base-rate $(or $(RATE),0.10) --true-lift $(or $(LIFT),0.08) \
		--seed $(or $(SEED),123) --output data/custom.csv

clean:  ## remove the virtualenv and test caches
	rm -rf $(VENV) .pytest_cache tests/__pycache__ src/__pycache__
