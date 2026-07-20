# Chinese Instrument Stem Separator (Demucs)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wushanyun64/chinese-instrument-demucs/blob/main/colab/chinese_instrument_demucs.ipynb)

A **single-target source separator** based on Demucs that isolates any Chinese
instrument (erhu, pipa, dizi, etc.) from arbitrary music mixtures.

## What it does

Takes any song → outputs an isolated instrument track.

## Design

- **Two-source model** (`<instrument>`, `other`) where `other = mixture − instrument`
- **Synthetic data pipeline** — mixtures built on-the-fly from isolated clips + instrument-free backgrounds
- **Warm-start from pretrained `htdemucs`** with reinitialized output head (4→2 sources)
- **Orchestrated via Dora + Hydra** for reproducible experiments

## Quickstart

```bash
# 1. Create environment + install
uv sync --extra dev

# 2. Build synthetic dataset (e.g. for erhu)
uv run python data_pipeline/build_dataset.py \
    --source-dir erhu_clips/ --bg-dir backgrounds/ --source-name erhu

# 3. Train
bash training/train.sh

# 4. Separate
uv run python inference/separate.py input.wav --sig <SIG> --stem erhu
```

## Useful commands

```bash
# Run tests
uv run pytest

# Build docs
uv run mkdocs serve
```

## Documentation

Full docs at [docs/index.md](docs/index.md) or run `uv run mkdocs serve`.

## Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wushanyun64/chinese-instrument-demucs/blob/main/colab/chinese_instrument_demucs.ipynb)

Click the badge above to run inference in your browser — no local setup needed.

## License

MIT
