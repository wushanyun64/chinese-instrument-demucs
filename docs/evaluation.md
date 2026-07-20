# Evaluation

## Metrics

We report **SI-SDR** and **SDR** (via `museval`) on a **held-out real test set** —
genuine songs with Chinese instrument, not synthetic data.

## Running

```bash
uv run python eval/evaluate.py --test-dir data/real_test/ --sig <SIG>
```

Prints per-track and mean instrument-stem SDR.

## Test set

The real test set is separate from the synthetic training/validation pipeline.
It measures generalization to real-world mixtures.
