.PHONY: env env-verify build-data validate-data train export separate eval test notebooks docs docs-serve clean

PYTHON := .venv/bin/python
UV := uv

env:
	$(UV) venv --python 3.12
	$(UV) pip install -e ".[dev]"
	@echo "✓ Environment ready — activate: source .venv/bin/activate"

env-verify:
	PYTHONPATH=vendor/demucs $(PYTHON) -c "import torch; print('CUDA available:', torch.cuda.is_available())"
	PYTHONPATH=vendor/demucs $(PYTHON) -c "import demucs; print('demucs import OK')"

build-data:
	PYTHONPATH=vendor/demucs $(PYTHON) data_pipeline/build_dataset.py

validate-data:
	PYTHONPATH=vendor/demucs $(PYTHON) data_pipeline/validate_contamination.py backgrounds/

train:
	bash training/train.sh

export:
	@echo "Usage: python -m tools.export SIG"

separate: inference/separate.py
	PYTHONPATH=vendor/demucs $(PYTHON) inference/separate.py

eval:
	PYTHONPATH=vendor/demucs $(PYTHON) eval/evaluate.py

test:
	PYTHONPATH=vendor/demucs $(PYTHON) -m pytest tests/ -v

notebooks:
	PYTHONPATH=vendor/demucs $(PYTHON) -m pytest --nbmake notebooks/ -v

docs:
	$(UV) run mkdocs build --strict

docs-serve:
	$(UV) run mkdocs serve

clean:
	rm -rf data/ outputs/ release_models/ metadata/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

clean-all: clean
	rm -rf .venv/
