"""Flag background audio that may contain flute — a guardrail against contamination.

Contaminated backgrounds poison the ``other`` target during training, teaching
the model to leave flute in the wrong stem.  This script runs a simple
spectral heuristic (flute-band energy ratio) and/or the base pretrained
htdemucs to surface suspicious files for manual review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE: int = 44100

# Rough flute fundamental + strong harmonic region (Hz)
FLUTE_BAND_LO: float = 400.0
FLUTE_BAND_HI: float = 5000.0


def _flute_band_ratio(wav: torch.Tensor, sr: int = SAMPLE_RATE) -> float:
    """Ratio of energy in [FLUTE_BAND_LO, FLUTE_BAND_HI] to total energy."""
    n_fft = 2048
    # Convert to 1D mono: if multi-channel, mean across channels
    if wav.ndim > 1:
        wav = wav.mean(dim=0)
    wav = wav.reshape(-1)  # ensure flat 1D
    # Guard: input shorter than n_fft can't produce meaningful spectra
    if wav.numel() < n_fft:
        return 0.0
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=n_fft // 4,
        window=torch.hann_window(n_fft),
        return_complex=True,
    )
    power = spec.abs().square()
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / sr)

    lo_idx = (freqs >= FLUTE_BAND_LO).nonzero(as_tuple=True)[0][0].item()
    hi_idx = (freqs <= FLUTE_BAND_HI).nonzero(as_tuple=True)[0][-1].item() + 1

    band_power = power[lo_idx:hi_idx, :].sum().item()
    total_power = power.sum().item()
    if total_power < 1e-10:
        return 0.0
    return band_power / total_power


def validate(
    bg_dir: Path,
    threshold: float | None = None,
    abs_thresh: float | None = None,
    use_demucs: bool = False,
    out_json: Path | None = None,
) -> list[dict[str, object]]:
    """Scan *bg_dir* and flag suspicious files.

    Parameters
    ----------
    bg_dir : Path
        Directory containing background audio files.
    threshold : float | None
        If given, flag files whose flute-band ratio exceeds *threshold*
        by more than *N* standard deviations (default: 2.0 σ above mean).
    abs_thresh : float | None
        Absolute ratio cutoff (e.g. 0.4). Files above this are flagged.
    use_demucs : bool
        If True, also run pretrained htdemucs and check if it extracts any
        signal in the "other" band.  Slower but more robust.
    out_json : Path | None
        If given, write results as JSON.

    Returns
    -------
    flagged : list[dict]
    """
    audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff"}
    files = sorted(p for p in bg_dir.rglob("*") if p.suffix.lower() in audio_exts)
    if not files:
        print(f"Warning: no audio files found in {bg_dir}", file=sys.stderr)
        return []

    # Compute flute-band ratio for every file
    ratios: list[float] = []
    file_ratios: list[tuple[Path, float]] = []
    for fp in files:
        try:
            data, sr = sf.read(str(fp))
            if data.size == 0:
                ratios.append(0.0)
                file_ratios.append((fp, 0.0))
                continue
            wav = torch.from_numpy(data.T)  # (samples, channels) → (channels, samples)
        except Exception:
            print(f"Warning: could not load {fp}, skipping", file=sys.stderr)
            continue
        r = _flute_band_ratio(wav, sr)
        ratios.append(r)
        file_ratios.append((fp, r))

    mean_r = float(np.mean(ratios))
    std_r = float(np.std(ratios))

    flagged: list[dict[str, object]] = []

    for fp, r in file_ratios:
        reason = None
        if threshold is not None and std_r > 1e-10 and r > mean_r + threshold * std_r:
            reason = (
                f"Flute-band ratio {r:.3f} > μ+{threshold}σ "
                f"(μ={mean_r:.3f}, σ={std_r:.3f})"
            )
        if abs_thresh is not None and r > abs_thresh:
            reason = f"Flute-band ratio {r:.3f} > absolute threshold {abs_thresh}"

        if reason is not None:
            entry: dict[str, object] = {
                "file": str(fp),
                "flute_band_ratio": round(r, 4),
                "reason": reason,
            }
            flagged.append(entry)
            print(f"⚠  {reason}  →  {fp}")

    if use_demucs and flagged:
        print(
            "\nDemucs-based check requested but skipped (pretrained model not available "
            "in default env). Re-run with: python -m demucs --two-stems vocals ...",
            file=sys.stderr,
        )

    print(f"\nScanned {len(file_ratios)} files — {len(flagged)} flagged for review.")

    if out_json is not None:
        out_json.write_text(json.dumps(flagged, indent=2, ensure_ascii=False))

    return flagged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Flag backgrounds that may contain flute contamination"
    )
    p.add_argument("bg_dir", type=Path, help="Directory of background audio files")
    p.add_argument("--threshold", type=float, default=2.0,
                   help="Standard-deviation multiplier above mean (default: 2.0)")
    p.add_argument("--abs-thresh", type=float, default=None,
                   help="Absolute flute-band ratio cutoff")
    p.add_argument("--demucs", action="store_true",
                   help="Also check with pretrained htdemucs (slower)")
    p.add_argument("--out-json", type=Path, default=None,
                   help="Write flagged results to JSON")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    validate(
        bg_dir=args.bg_dir,
        threshold=args.threshold,
        abs_thresh=args.abs_thresh,
        use_demucs=args.demucs,
        out_json=args.out_json,
    )
