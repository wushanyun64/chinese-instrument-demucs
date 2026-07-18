# Inference

## CLI (via vendored demucs)

```bash
python -m demucs --repo ./release_models -n SIG --two-stems chinese-flute input.wav
```

Outputs: `chinese-flute.wav` and `no_chinese-flute.wav`. Keep the flute stem.

## Python wrapper

```bash
python inference/separate.py input.wav          # single file
python inference/separate.py input_folder/       # batch folder
```

The wrapper handles resampling, mono→stereo upmixing, and batch processing.
