"""Audio utilities for the training dataset synthesis pipeline.

Provides loudness normalization, resampling, stereo upmixing, SNR-based
mixing, and light augmentation — all operating on torch tensors at 44.1 kHz."""

from __future__ import annotations

import math
from pathlib import Path

import julius
import numpy as np
import soundfile as sf
import torch


SAMPLE_RATE: int = 44100


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_audio(path: str | Path, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load an audio file, resampling to *target_sr* (default 44.1 kHz).

    Returns a ``(channels, samples)`` float tensor.
    """
    data, sr = sf.read(str(path))
    wav = torch.from_numpy(data.T).float()  # (samples, channels) → (channels, samples)
    if sr != target_sr:
        wav = julius.resample_frac(wav, sr, target_sr)
    return wav


def get_audio_info(path: str | Path) -> tuple[int, int]:
    """Return ``(sample_rate, total_frames)`` without loading audio data."""
    info = sf.info(str(path))
    return info.samplerate, info.frames


def load_audio_segment(
    path: str | Path,
    offset: int = 0,
    frames: int | None = None,
    target_sr: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Load a slice of an audio file.

    Parameters
    ----------
    path : Path
        Audio file.
    offset : int
        Start frame (0-indexed) in the file's native sample rate.
    frames : int | None
        Number of frames to read.  ``None`` = read to end.
    target_sr : int
        Resample to this rate (default 44.1 kHz).

    Returns
    -------
    ``(channels, samples)`` float tensor.
    """
    data, sr = sf.read(str(path), start=offset, stop=None if frames is None else offset + frames)
    wav = torch.from_numpy(data.T).float()
    if sr != target_sr:
        wav = julius.resample_frac(wav, sr, target_sr)
    return wav


def save_audio(wav: torch.Tensor, path: str | Path, sr: int = SAMPLE_RATE) -> None:
    """Save a ``(channels, samples)`` float tensor to a WAV file (FLOAT subtype)."""
    sf.write(str(path), wav.cpu().numpy().T, sr, subtype="FLOAT")


# ---------------------------------------------------------------------------
# Format normalization
# ---------------------------------------------------------------------------

def ensure_stereo(wav: torch.Tensor) -> torch.Tensor:
    """Return a stereo tensor.

    Mono → duplicate channel.  Multichannel > 2 → take first two channels.
    """
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) == 1:
        wav = wav.repeat(2, 1)
    elif wav.size(0) > 2:
        wav = wav[:2, :]
    return wav


def ensure_length(wav: torch.Tensor, target_samples: int) -> torch.Tensor:
    """Crop or (naively) loop-pad *wav* to exactly *target_samples*."""
    ch, n = wav.shape
    if n >= target_samples:
        return wav[:, :target_samples]
    repeats = math.ceil(target_samples / n)
    wav = wav.repeat(1, repeats)
    return wav[:, :target_samples]


# ---------------------------------------------------------------------------
# Loudness / SNR
# ---------------------------------------------------------------------------

def rms(wav: torch.Tensor) -> torch.Tensor:
    """Root-mean-square (scalar) of a waveform."""
    return wav.square().mean().sqrt()


def loudness_normalize(wav: torch.Tensor, target_db: float = -20.0) -> torch.Tensor:
    """Scale *wav* so its RMS corresponds to *target_db* relative to full-scale.

    Returns the scaled waveform (not a copy — the input is modified).
    """
    current_rms = rms(wav)
    if current_rms < 1e-10:
        return wav  # silence
    target_linear = 10 ** (target_db / 20.0)
    scale = target_linear / current_rms
    return wav * scale


def mix_at_snr(
    signal: torch.Tensor, other: torch.Tensor, snr_db: float
) -> torch.Tensor:
    """Mix *signal* and *other* at the given signal-to-noise ratio.

    Parameters
    ----------
    signal : (channels, samples)
        The "signal" whose power defines the reference.
    other : (channels, samples)
        The "noise" component (must have the same shape as *signal*).
    snr_db : float
        Desired SNR in dB.  Positive = signal louder; negative = other louder.

    Returns
    -------
    mixture : (channels, samples)
        ``mixture = signal + scaled_other`` where ``scaled_other`` is adjusted
        so the power ratio matches *snr_db*.
    """
    p_signal = signal.square().mean()
    p_noise = other.square().mean()
    if p_noise < 1e-10:
        return signal + other  # silence background
    # SNR = 10 * log10(P_signal / (a^2 * P_noise))
    # => a = sqrt(P_signal / (P_noise * 10^(SNR/10)))
    a_squared = p_signal / (p_noise * (10 ** (snr_db / 10.0)))
    scale = a_squared.sqrt()
    return signal + scale * other


# ---------------------------------------------------------------------------
# Light augmentation (per the plan: modest, Demucs does heavier remix later)
# ---------------------------------------------------------------------------

def pitch_shift(wav: torch.Tensor, n_steps: int = 0) -> torch.Tensor:
    """Pitch-shift by *n_steps* semitones using julius.

    Zero-pads the signal before resampling to avoid boundary artifacts;
    trims back to the original length afterwards.
    """
    if n_steps == 0:
        return wav
    rate = 2 ** (-n_steps / 12.0)
    # pad to avoid boundary clipping
    pad = int(0.1 * wav.size(-1))
    padded = torch.nn.functional.pad(wav, (pad, pad))
    shifted = julius.resample_frac(padded, int(SAMPLE_RATE * rate), SAMPLE_RATE)
    start = int(pad * rate)
    end = start + wav.size(-1)
    return shifted[..., start:end]


def time_stretch(wav: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
    """Time-stretch by *rate* (1.0 = no change) using julius resample."""
    if rate == 1.0:
        return wav
    stretched = julius.resample_frac(
        wav, int(SAMPLE_RATE * rate), SAMPLE_RATE
    )
    return stretched


def random_gain(wav: torch.Tensor, db_range: tuple[float, float] = (-6.0, 6.0)) -> torch.Tensor:
    """Apply a uniform-random gain within *db_range* (min, max) dB."""
    lo, hi = db_range
    db = float(torch.empty(1).uniform_(lo, hi))
    return wav * (10 ** (db / 20.0))
