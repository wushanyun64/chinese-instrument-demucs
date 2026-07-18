# Chinese Flute Demucs — Documentation

## What this project does

Extracts Chinese flute (dizi, xiao, bawu) stems from arbitrary music mixtures
using a fine-tuned Demucs model.

## Quickstart

```bash
make env && conda activate chinese-flute-demucs
make build-data
make train
make separate INPUT=input.wav
```

## How it works

- **Single-target, two-source design:** `['chinese-flute', 'other']` where `other = mixture − flute`
- **Synthetic data:** mixtures built on-the-fly from isolated flute recordings + flute-free backgrounds
- **Warm-start from pretrained htdemucs:** transfers learned audio representations; reinitializes output head (4→2 sources)

## Where to go next

- [Installation](installation.md) — set up the environment
- [Concepts](concepts.md) — why it's designed this way
- [Data](data.md) — preparing your audio
- [Training](training.md) — launching and monitoring training
- [Inference](inference.md) — using the trained model
- [Evaluation](evaluation.md) — measuring performance
