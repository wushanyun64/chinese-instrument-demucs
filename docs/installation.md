# Installation

## Prerequisites

- NVIDIA GPU with CUDA 11.8+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

## Setup

```bash
# Create environment + install all dependencies
uv sync

# Verify
uv run --env PYTHONPATH=vendor/demucs python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
uv run --env PYTHONPATH=vendor/demucs python -c "import demucs; print('demucs import OK')"
```

Expected output:
```
CUDA available: True
demucs import OK
```

## Vendored Demucs

This repo vendors `facebookresearch/demucs` (archived 2025-01-01) directly in `vendor/demucs/`.
It is imported as a local package; no separate `pip install demucs` is needed.

Commands that need demucs must set `PYTHONPATH=vendor/demucs`:

```bash
uv run --env PYTHONPATH=vendor/demucs python ...
```

## Optional: Makefile

A `Makefile` is included for convenience (`make test`, `make docs`, `make clean`),
but all commands work directly with `uv run`.
