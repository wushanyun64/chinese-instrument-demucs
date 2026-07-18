# Installation

## Prerequisites

- NVIDIA GPU with CUDA 11.8+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

## Setup

```bash
# Create environment + install all deps
make env

# Activate (or use uv run <command>)
source .venv/bin/activate

# Verify
make env-verify
```

Expected output:
```
CUDA available: True
demucs import OK
```

## Manual setup (without make)

```bash
uv venv
uv pip install -e ".[dev]"
```

## Vendored Demucs

This repo vendors `facebookresearch/demucs` (archived 2025-01-01) directly in `vendor/demucs/`.
It is imported as a local package; no separate `pip install demucs` is needed.

All commands set `PYTHONPATH=vendor/demucs` so the vendored package is discoverable.

## Running without activating the venv

```bash
uv run --with-editable . python -c "import torch; print(torch.cuda.is_available())"
```
