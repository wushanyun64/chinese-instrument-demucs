# Data

## Input requirements

### Instrument clips (`source_clips/`)

Isolated Chinese instrument recordings. **Diversity is the priority:**
- Multiple instruments: dizi, xiao, bawu, erhu, pipa, etc.
- Multiple players, keys, articulations
- Various recording conditions

### Backgrounds (`backgrounds/`)

Music containing **no target instrument** — any genre. Sources: MUSDB18-HQ mixtures,
royalty-free libraries, own collection.

## Output format

```
data/instrument_dataset/
├── train/
│   ├── 000001/
│   │   ├── mixture.wav
│   │   ├── <instrument>.wav
│   │   └── other.wav
│   └── ...
└── valid/
    └── 000501/ ...
```

All audio: **44.1 kHz, stereo, float**.

## Synthesis pipeline

See `data_pipeline/build_dataset.py` for the full pipeline. Key steps:
1. Sample instrument + background clips; take a common-length segment (7–11 s)
2. Loudness-normalize each, then mix at randomized SNR (−5 to +10 dB)
3. Write `<instrument>.wav`, `other.wav`, `mixture.wav = instrument + other`

Augmentation (remix/shift/pitch/gain) is applied by Demucs at train time
(see `configs/variant/instrument_ft.yaml`), not baked into the dataset.

## Guardrails

- Train/valid split by **source clip identity** (prevents leakage)
- Background verification pass flags instrument-contaminated files (`validate_contamination.py`)
