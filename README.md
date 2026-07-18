# Chinese Instrument Stem Separator (Demucs)

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
uv sync

# 2. Build synthetic dataset (e.g. for erhu)
uv run --env PYTHONPATH=vendor/demucs python data_pipeline/build_dataset.py \
    --source-dir erhu_clips/ --bg-dir backgrounds/ --source-name erhu

# 3. Train
bash training/train.sh

# 4. Separate
uv run --env PYTHONPATH=vendor/demucs python inference/separate.py input.wav --sig <SIG> --stem erhu
```

## Useful commands

```bash
# Run tests
uv run --env PYTHONPATH=vendor/demucs pytest

# Build docs
uv run mkdocs serve
```

## Documentation

Full docs at [docs/index.md](docs/index.md) or run `uv run mkdocs serve`.

## License

MIT
