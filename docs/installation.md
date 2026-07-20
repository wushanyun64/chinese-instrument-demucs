# Installation

## Google Colab (no setup required)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wushanyun64/chinese-instrument-demucs/blob/main/colab/chinese_instrument_demucs.ipynb)

Run inference directly in your browser with a free T4 GPU — nothing to install.

## Prerequisites

- NVIDIA GPU with CUDA 11.8+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

## Setup

```bash
# Create environment + install all dependencies (including dev tools)
uv sync --extra dev

# Verify
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
uv run python -c "import demucs; print('demucs import OK')"
```

Expected output:
```
CUDA available: True
demucs import OK
```

## Demucs

This project uses the official [adefossez/demucs](https://github.com/adefossez/demucs) package
from PyPI. Training, inference, and evaluation all work with `pip install "demucs[train]"` —
no vendored copy needed. The `[train]` extra pulls in Dora, Hydra, and all training dependencies.

Dora base configs are mirrored locally at `configs/demucs_base/` since the pip wheel doesn't
ship the `conf/` directory.

## Optional: Makefile

A `Makefile` is included for convenience (`make test`, `make docs`, `make clean`),
but all commands work directly with `uv run`.
