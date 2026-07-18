r"""Synthesize a ``[chinese-flute, other]`` wav dataset for Demucs training.

Builds mixtures by overlaying isolated flute recordings onto flute-free
background music:

    mixture = gain_f * flute + gain_o * other

Output format (Demucs wav dataset convention)::

    data/flute_dataset/
    ├── train/
    │   ├── 000001/
    │   │   ├── mixture.wav
    │   │   ├── chinese-flute.wav
    │   │   └── other.wav
    │   └── ...
    └── valid/
        └── 000501/ ...

All audio: **44.1 kHz, stereo, float**.

Usage::

    python data_pipeline/build_dataset.py \
        --flute-dir flute_clips/ \
        --bg-dir backgrounds/ \
        --out data/flute_dataset \
        --num-train 500 --num-valid 50 \
        --seg-len 8 --snr-min -5 --snr-max 10
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import numpy as np
import torch

from data_pipeline.audio_utils import (
    SAMPLE_RATE,
    ensure_length,
    ensure_stereo,
    load_audio,
    loudness_normalize,
    mix_at_snr,
    pitch_shift,
    random_gain,
    save_audio,
    time_stretch,
)


def collect_audio_files(directory: Path, extensions: set[str] | None = None) -> list[Path]:
    """Return a sorted list of audio file paths in *directory*.

    By default accepts ``.wav``, ``.flac``, ``.mp3``, ``.ogg``.
    """
    if extensions is None:
        extensions = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff"}
    files: list[Path] = []
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in extensions:
            files.append(p)
    return files


def build_track(
    flute_file: Path,
    bg_file: Path,
    seg_samples: int,
    snr_db: float,
    augment_flute: bool = True,
    augment_bg: bool = False,
    flute_gain_db: float = -20.0,
    bg_gain_db: float = -20.0,
) -> dict[str, torch.Tensor]:
    """Build a single ``{mixture, chinese-flute, other}`` track.

    Parameters
    ----------
    flute_file : Path
        Isolated flute recording.
    bg_file : Path
        Flute-free background music.
    seg_samples : int
        Number of samples for the output segment.
    snr_db : float
        Target SNR (positive = flute louder).
    augment_flute, augment_bg : bool
        Whether to apply light pitch/time/gain augmentation.
    flute_gain_db, bg_gain_db : float
        Reference loudness targets before mixing.

    Returns
    -------
    dict with keys ``"chinese-flute"``, ``"other"``, ``"mixture"`` —
    each a ``(2, seg_samples)`` float tensor.
    """
    # Load & normalize format
    flute = ensure_stereo(load_audio(flute_file))
    bg = ensure_stereo(load_audio(bg_file))

    # Take a random common-length segment
    min_len = min(flute.size(-1), bg.size(-1))
    if min_len < seg_samples:
        flute = ensure_length(flute, seg_samples)
        bg = ensure_length(bg, seg_samples)
        start_frame = 0
    else:
        start_frame = random.randint(0, min_len - seg_samples)
        flute = flute[:, start_frame : start_frame + seg_samples]
        bg = bg[:, start_frame : start_frame + seg_samples]

    # Loudness normalize each to a consistent reference
    flute = loudness_normalize(flute, target_db=flute_gain_db)
    bg = loudness_normalize(bg, target_db=bg_gain_db)

    # Light augmentation (Demucs does heavier remix at train time)
    if augment_flute:
        flute = pitch_shift(flute, n_steps=random.choice([0, 0, 0, -2, 2]))
        if random.random() < 0.3:
            flute = time_stretch(flute, rate=random.uniform(0.9, 1.1))
        flute = random_gain(flute, db_range=(-3.0, 3.0))
    if augment_bg:
        bg = random_gain(bg, db_range=(-3.0, 3.0))

    # Ensure common length after augmentation
    flute = ensure_length(flute, seg_samples)
    bg = ensure_length(bg, seg_samples)

    # Mix
    mixture = mix_at_snr(flute, bg, snr_db)

    # The plan's invariant: mixture == chinese-flute + other sample-exact.
    # "other" must be the ACTUAL background component in the mixture,
    # not the raw background.  Recover it from the mixture.
    other_component = mixture - flute

    return {
        "chinese-flute": flute,
        "other": other_component,
        "mixture": mixture,
    }


def build_dataset(
    flute_dir: Path,
    bg_dir: Path,
    out_dir: Path,
    num_train: int = 500,
    num_valid: int = 50,
    seg_len: float = 8.0,
    snr_min: float = -5.0,
    snr_max: float = 10.0,
    flute_gain_db: float = -20.0,
    bg_gain_db: float = -20.0,
    seed: int = 42,
) -> None:
    """Main synthesis entry-point.

    Split is by **source clip identity**: each flute file contributes to
    either train XOR valid, not both.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    seg_samples = int(seg_len * SAMPLE_RATE)

    flute_files = collect_audio_files(flute_dir)
    bg_files = collect_audio_files(bg_dir)

    if len(flute_files) < 2:
        raise RuntimeError(f"Need ≥ 2 flute clips; found {len(flute_files)}")
    if len(bg_files) < 2:
        raise RuntimeError(f"Need ≥ 2 background clips; found {len(bg_files)}")

    # Split flute files by identity
    random.shuffle(flute_files)
    split = max(1, len(flute_files) // 5)
    train_flutes = flute_files[split:]
    valid_flutes = flute_files[:split]

    random.shuffle(bg_files)
    train_bgs = bg_files[split:]
    valid_bgs = bg_files[:split]

    print(
        f"Flute clips: {len(flute_files)} total → "
        f"{len(train_flutes)} train / {len(valid_flutes)} valid"
    )
    print(
        f"Backgrounds: {len(bg_files)} total → "
        f"{len(train_bgs)} train / {len(valid_bgs)} valid"
    )

    for split_name, flutes, bgs, n_tracks in [
        ("train", train_flutes, train_bgs, num_train),
        ("valid", valid_flutes, valid_bgs, num_valid),
    ]:
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        flute_cycle = itertools.cycle(flutes)
        bg_cycle = itertools.cycle(bgs)

        for idx in range(n_tracks):
            track_dir = split_dir / f"{idx + 1:06d}"
            track_dir.mkdir(exist_ok=True)

            snr_db = random.uniform(snr_min, snr_max)
            stems = build_track(
                flute_file=next(flute_cycle),
                bg_file=next(bg_cycle),
                seg_samples=seg_samples,
                snr_db=snr_db,
                flute_gain_db=flute_gain_db,
                bg_gain_db=bg_gain_db,
            )

            # Write stems (order: flute, other, mixture with sample-exact sum guarantee)
            for name, wav in stems.items():
                save_audio(wav, track_dir / f"{name}.wav")

            if (idx + 1) % 100 == 0:
                print(f"  {split_name}: {idx + 1}/{n_tracks} tracks built")

        print(f"  {split_name}: {n_tracks}/{n_tracks} tracks built")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthesize Chinese-flute Demucs dataset")
    p.add_argument("--flute-dir", required=True, type=Path, help="Isolated flute clips")
    p.add_argument("--bg-dir", required=True, type=Path, help="Flute-free background music")
    p.add_argument("--out", default=Path("data/flute_dataset"), type=Path)
    p.add_argument("--num-train", default=500, type=int)
    p.add_argument("--num-valid", default=50, type=int)
    p.add_argument("--seg-len", default=8.0, type=float, help="Segment length in seconds")
    p.add_argument("--snr-min", default=-5.0, type=float)
    p.add_argument("--snr-max", default=10.0, type=float)
    p.add_argument("--flute-gain-db", default=-20.0, type=float)
    p.add_argument("--bg-gain-db", default=-20.0, type=float)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        flute_dir=args.flute_dir,
        bg_dir=args.bg_dir,
        out_dir=args.out,
        num_train=args.num_train,
        num_valid=args.num_valid,
        seg_len=args.seg_len,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        flute_gain_db=args.flute_gain_db,
        bg_gain_db=args.bg_gain_db,
        seed=args.seed,
    )
