# Inference

## CLI (via vendored demucs)

```bash
uv run --env PYTHONPATH=vendor/demucs python -m demucs --repo ./release_models -n SIG --two-stems chinese-instrument input.wav
```

Outputs: `chinese-instrument.wav` and `no_chinese-instrument.wav`. Keep the instrument stem.

## Python wrapper

```bash
# Single file
uv run --env PYTHONPATH=vendor/demucs python inference/separate.py input.wav --sig <SIG> --stem erhu

# Batch folder
uv run --env PYTHONPATH=vendor/demucs python inference/separate.py input_folder/ --sig <SIG> --stem erhu
```

The wrapper handles resampling, mono→stereo upmixing, and batch processing.
