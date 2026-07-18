# Chinese Flute Stem Separator (Demucs)

A **single-target source separator** based on Demucs that extracts Chinese flute
(dizi, xiao, bawu) stems from arbitrary music mixtures.

## What it does

Takes any song → outputs an isolated Chinese flute track.

## Design

- **Two-source model** (`chinese-flute`, `other`) where `other = mixture − flute`
- **Synthetic data pipeline** — mixtures built on-the-fly from isolated flute clips + flute-free backgrounds
- **Warm-start from pretrained `htdemucs`** with reinitialized output head (4→2 sources)
- **Orchestrated via Dora + Hydra** for reproducible experiments

## Quickstart

```bash
# 1. Environment
make env
conda activate chinese-flute-demucs

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
