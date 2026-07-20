# Chinese Instrument Demucs — Documentation

## What this project does

Extracts a target Chinese instrument (erhu, pipa, dizi, etc.) from arbitrary music mixtures
using a fine-tuned Demucs model.

## Quickstart

```bash
# 1. Create environment + install
uv sync

# 2. Build synthetic dataset (e.g. for erhu)
uv run python data_pipeline/build_dataset.py \
    --source-dir erhu_clips/ --bg-dir backgrounds/ --source-name erhu

# 3. Train
bash training/train.sh

# 4. Separate
uv run python inference/separate.py input.wav --sig <SIG> --stem erhu
```

## How it works

- **Single-target, two-source design:** `['<instrument>', 'other']` where `other = mixture − instrument`
- **Synthetic data:** mixtures built on-the-fly from isolated instrument recordings + instrument-free backgrounds
- **Warm-start from pretrained htdemucs:** transfers learned audio representations; reinitializes output head (4→2 sources)

## Where to go next

- [Installation](installation.md) — set up the environment
- [Concepts](concepts.md) — why it's designed this way
- [Data](data.md) — preparing your audio
- [Training](training.md) — launching and monitoring training
- [Inference](inference.md) — using the trained model
- [Evaluation](evaluation.md) — measuring performance
