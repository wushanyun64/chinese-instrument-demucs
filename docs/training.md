# Training

## Launch

```bash
dora run -d model=htdemucs dset=flute variant=flute_ft
```

`-d` uses all GPUs. Capture the resulting **experiment signature (SIG)**.

## Checkpoints

Training checkpoints are saved to `outputs/<SIG>/`. Resume:

```bash
dora run -d model=htdemucs dset=flute variant=flute_ft continue_from=<SIG>
```

## Rescan / cleanup

```bash
# Force rescan of experiment metadata
rm -rf metadata/

# Clear a bad run
dora run --clear ...
```

## Hardware guidance

Adjust in `configs/variant/flute_ft.yaml`:
- `dset.segment`: reduce for lower VRAM
- `dset.batch_size`: reduce for lower VRAM
- Model channels: reduce if VRAM-constrained
