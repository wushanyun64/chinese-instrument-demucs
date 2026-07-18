# Installation

## Prerequisites

- NVIDIA GPU with CUDA 11.8+
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommended)

## Setup

```bash
# Create environment
make env
conda activate chinese-flute-demucs

# Verify
make env-verify
```

Expected output:
```
CUDA available: True
demucs version: ...
```

## Vendored Demucs

This repo vendors `facebookresearch/demucs` (archived 2025-01-01) as a git submodule:

```bash
git submodule update --init --recursive
```

The vendored demucs is imported as a local package; no separate `pip install demucs` is needed.
