# Data

## Input requirements

### Flute clips (`flute_clips/`)

Isolated Chinese flute recordings. **Diversity is the priority:**
- Multiple instruments: dizi, xiao, bawu
- Multiple players, keys, articulations
- Various recording conditions
- Include the dizi's buzzing membrane (dimo) resonance

### Backgrounds (`backgrounds/`)

Music containing **no flute** — any genre. Sources: MUSDB18-HQ mixtures,
royalty-free libraries, own collection.

## Output format

```
data/flute_dataset/
├── train/
│   ├── 000001/
│   │   ├── mixture.wav
│   │   ├── chinese-flute.wav
│   │   └── other.wav
│   └── ...
└── valid/
    └── 000501/ ...
```

All audio: **44.1 kHz, stereo, float**.

## Synthesis pipeline

See `data_pipeline/build_dataset.py` for the full pipeline. Key steps:
1. Sample flute + background clips; take a common-length segment (7–11 s)
2. Loudness-normalize each, then mix at randomized SNR (−5 to +10 dB)
3. Apply light augmentation (pitch shift, time stretch, random gain)
4. Write `chinese-flute.wav`, `other.wav`, `mixture.wav = flute + other`

## Guardrails

- Train/valid split by **source clip identity** (prevents leakage)
- Background verification pass flags flute-contaminated files (`validate_flute_free.py`)
