# Chinese Instrument Stem Separator (Demucs)

A **single-target source separator** based on Demucs that extracts Chinese instrument
(dizi, xiao, bawu) stems from arbitrary music mixtures.

## What it does

Takes any song → outputs an isolated Chinese instrument track.

## Design

- **Two-source model** (`chinese-instrument`, `other`) where `other = mixture − target instrument`
- **Synthetic data pipeline** — mixtures built on-the-fly from isolated target instrument clips + target instrument-free backgrounds
- **Warm-start from pretrained `htdemucs`** with reinitialized output head (4→2 sources)
- **Orchestrated via Dora + Hydra** for reproducible experiments

## Quickstart

```bash
# 1. Environment (uses uv)
make env
source .venv/bin/activate

# 2. Build synthetic dataset
make build-data

# 3. Train
make train

# 4. Separate
make separate INPUT=input.wav
```

## Documentation

Full docs at [docs/index.md](docs/index.md) or run `make docs-serve`.

## License

MIT
