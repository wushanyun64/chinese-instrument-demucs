# Evaluation

## Metrics

We report **SI-SDR** and **SDR** (via `museval`) on a **held-out real test set** —
genuine songs with Chinese flute, not synthetic data.

## Running

```bash
make eval
```

Prints per-track and mean flute-stem SDR.

## Test set

The real test set is separate from the synthetic training/validation pipeline.
It measures generalization to real-world mixtures.
