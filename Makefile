.PHONY: env build-data train export separate eval test notebooks docs docs-serve clean

PYTHON := python3
PIP := pip

env:
	conda env create -f environment-cuda.yml || conda env update -f environment-cuda.yml
	@echo "Run: conda activate chinese-flute-demucs"

env-verify:
	$(PYTHON) -c "import torch, torchaudio; print('CUDA available:', torch.cuda.is_available())"
	$(PYTHON) -c "import demucs; print('demucs version:', demucs.__version__)"

build-data:
	$(PYTHON) data_pipeline/build_dataset.py

validate-data:
	$(PYTHON) data_pipeline/validate_flute_free.py backgrounds/

train:
	bash training/train.sh

export:
	@echo "Usage: python -m tools.export SIG"

separate: inference/separate.py
	$(PYTHON) inference/separate.py

eval:
	$(PYTHON) eval/evaluate.py

test:
	$(PYTHON) -m pytest tests/ -v

notebooks:
	$(PYTHON) -m pytest --nbmake notebooks/ -v

docs:
	mkdocs build --strict

docs-serve:
	mkdocs serve

clean:
	rm -rf data/ outputs/ release_models/ metadata/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

clean-all: clean
	conda env remove -n chinese-flute-demucs || true
