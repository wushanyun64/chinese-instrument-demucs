# Sample Data

This directory holds tiny, license-clear audio clips so notebooks and the
quickstart run with zero external downloads.

## Contents (to be added)

- `flute_sample.wav` — a short isolated Chinese flute clip
- `bg_sample.wav` — a short flute-free background clip
- `mixture_sample.wav` — a short real-mixture test clip

## Creating synthetic test clips

Until real recordings are added, the test suite generates synthetic clips
(sine tones + white noise) in temporary directories via pytest fixtures.
