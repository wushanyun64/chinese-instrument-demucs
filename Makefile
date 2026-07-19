.PHONY: env env-verify build-data validate-data train export separate eval test notebooks docs docs-serve clean

UV := uv

# ---- Primary workflow: use uv directly ----
#    uv sync                        # create env + install
#    uv run --env PYTHONPATH=vendor/demucs pytest
#    uv run --env PYTHONPATH=vendor/demucs python data_pipeline/build_dataset.py ...
#    bash training/train.sh
#    uv run --env PYTHONPATH=vendor/demucs python inference/separate.py ...
#
# These Makefile targets are convenience wrappers.

env:
	$(UV) sync
	@echo "Ready — use: uv run --env PYTHONPATH=vendor/demucs ..."

env-verify:
	$(UV) run --env PYTHONPATH=vendor/demucs python -c "import torch; print('CUDA:', torch.cuda.is_available())"
	$(UV) run --env PYTHONPATH=vendor/demucs python -c "import demucs; print('demucs OK')"

build-data:
	$(UV) run --env PYTHONPATH=vendor/demucs python data_pipeline/build_dataset.py

validate-data:
	$(UV) run --env PYTHONPATH=vendor/demucs python data_pipeline/validate_contamination.py backgrounds/

train:
	bash training/train.sh

separate:
	$(UV) run --env PYTHONPATH=vendor/demucs python inference/separate.py

eval:
	$(UV) run --env PYTHONPATH=vendor/demucs python eval/evaluate.py

test:
	$(UV) run --env PYTHONPATH=vendor/demucs pytest tests/ -v

notebooks:
	$(UV) run --env PYTHONPATH=vendor/demucs pytest --nbmake notebooks/ -v

docs:
	$(UV) run mkdocs build --strict

docs-serve:
	$(UV) run mkdocs serve

clean:
	rm -rf data/ outputs/ release_models/ metadata/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

clean-all: clean
	rm -rf .venv/
