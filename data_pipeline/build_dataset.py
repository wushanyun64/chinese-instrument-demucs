r"""Synthesize a ``[<instrument>, other]`` wav dataset for Demucs training.

Builds mixtures by overlaying isolated instrument recordings onto
instrument-free background music:

    mixture = gain_s * source + gain_o * other

Output format (Demucs wav dataset convention)::

    data/instrument_dataset/
    ├── train/
    │   ├── 000001/
    │   │   ├── mixture.wav
    │   │   ├── <instrument>.wav
    │   │   └── other.wav
    │   └── ...
    └── valid/
        └── 000501/ ...

All audio: **44.1 kHz, stereo, float**.

Usage::

    python data_pipeline/build_dataset.py \
        --source-dir erhu_clips/ \
        --bg-dir backgrounds/ \
        --source-name erhu \
        --out data/instrument_dataset \
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
    get_audio_info,
    load_audio,
    load_audio_segment,
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
    source_file: Path,
    bg_file: Path,
    seg_samples: int,
    snr_db: float,
    source_name: str = "chinese-instrument",
    augment_source: bool = True,
    augment_bg: bool = False,
    source_gain_db: float = -20.0,
    bg_gain_db: float = -20.0,
) -> dict[str, torch.Tensor]:
    """Build a single ``{mixture, <source_name>, other}`` track.

    Parameters
    ----------
    source_file : Path
        Isolated instrument recording.
    bg_file : Path
        Instrument-free background music.
    seg_samples : int
        Number of samples for the output segment.
    snr_db : float
        Target SNR (positive = source louder).
    source_name : str
        Name of the target instrument (used as stem filename).
    augment_source, augment_bg : bool
        Whether to apply light pitch/time/gain augmentation.
    source_gain_db, bg_gain_db : float
        Reference loudness targets before mixing.

    Returns
    -------
    dict with keys ``source_name``, ``"other"``, ``"mixture"`` —
    each a ``(2, seg_samples)`` float tensor.
    """
    # Probe durations without loading audio data
    src_sr, src_frames = get_audio_info(source_file)
    bg_sr, bg_frames = get_audio_info(bg_file)

    # Convert to target sample rate to determine available segment windows
    src_len = int(src_frames * SAMPLE_RATE / src_sr)
    bg_len = int(bg_frames * SAMPLE_RATE / bg_sr)

    # Pick a random start offset and load only the segment needed
    min_len = min(src_len, bg_len)
    if min_len < seg_samples:
        # Clip(s) too short — load full file and loop-pad
        source = ensure_stereo(load_audio(source_file))
        bg = ensure_stereo(load_audio(bg_file))
    else:
        start_frame = random.randint(0, min_len - seg_samples)
        # Convert frame offset back to each file's native sample rate
        src_offset = int(start_frame * src_sr / SAMPLE_RATE)
        bg_offset = int(start_frame * bg_sr / SAMPLE_RATE)
        src_seg_frames = int(seg_samples * src_sr / SAMPLE_RATE)
        bg_seg_frames = int(seg_samples * bg_sr / SAMPLE_RATE)
        source = ensure_stereo(load_audio_segment(source_file, offset=src_offset, frames=src_seg_frames))
        bg = ensure_stereo(load_audio_segment(bg_file, offset=bg_offset, frames=bg_seg_frames))

    # Trim to exact length
    source = ensure_length(source, seg_samples)
    bg = ensure_length(bg, seg_samples)

    # Loudness normalize each to a consistent reference
    source = loudness_normalize(source, target_db=source_gain_db)
    bg = loudness_normalize(bg, target_db=bg_gain_db)

    # Light augmentation (Demucs does heavier remix at train time)
    if augment_source:
        source = pitch_shift(source, n_steps=random.choice([0, 0, 0, -2, 2]))
        if random.random() < 0.3:
            source = time_stretch(source, rate=random.uniform(0.9, 1.1))
        source = random_gain(source, db_range=(-3.0, 3.0))
    if augment_bg:
        bg = random_gain(bg, db_range=(-3.0, 3.0))

    # Ensure common length after augmentation
    source = ensure_length(source, seg_samples)
    bg = ensure_length(bg, seg_samples)

    # Mix
    mixture = mix_at_snr(source, bg, snr_db)

    # The plan's invariant: mixture == <source> + other sample-exact.
    # "other" must be the ACTUAL background component in the mixture.
    other_component = mixture - source

    return {
        source_name: source,
        "other": other_component,
        "mixture": mixture,
    }


def build_dataset(
    source_dir: Path,
    bg_dir: Path,
    out_dir: Path,
    source_name: str = "chinese-instrument",
    num_train: int = 500,
    num_valid: int = 50,
    seg_len: float = 8.0,
    snr_min: float = -5.0,
    snr_max: float = 10.0,
    source_gain_db: float = -20.0,
    bg_gain_db: float = -20.0,
    augment_source: bool = True,
    augment_bg: bool = False,
    seed: int = 42,
) -> None:
    """Main synthesis entry-point.

    Split is by **source clip identity**: each source file contributes to
    either train XOR valid, not both.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    seg_samples = int(seg_len * SAMPLE_RATE)

    source_files = collect_audio_files(source_dir)
    bg_files = collect_audio_files(bg_dir)

    if len(source_files) < 2:
        raise RuntimeError(f"Need ≥ 2 source clips; found {len(source_files)}")
    if len(bg_files) < 2:
        raise RuntimeError(f"Need ≥ 2 background clips; found {len(bg_files)}")

    # Split source files by identity
    random.shuffle(source_files)
    split = max(1, len(source_files) // 5)
    train_sources = source_files[split:]
    valid_sources = source_files[:split]

    random.shuffle(bg_files)
    train_bgs = bg_files[split:]
    valid_bgs = bg_files[:split]

    print(
        f"Source clips ({source_name}): {len(source_files)} total → "
        f"{len(train_sources)} train / {len(valid_sources)} valid"
    )
    print(
        f"Backgrounds: {len(bg_files)} total → "
        f"{len(train_bgs)} train / {len(valid_bgs)} valid"
    )

    for split_name, sources, bgs, n_tracks in [
        ("train", train_sources, train_bgs, num_train),
        ("valid", valid_sources, valid_bgs, num_valid),
    ]:
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        source_cycle = itertools.cycle(sources)
        bg_cycle = itertools.cycle(bgs)

        for idx in range(n_tracks):
            track_dir = split_dir / f"{idx + 1:06d}"
            track_dir.mkdir(exist_ok=True)

            snr_db = random.uniform(snr_min, snr_max)
            stems = build_track(
                source_file=next(source_cycle),
                bg_file=next(bg_cycle),
                seg_samples=seg_samples,
                snr_db=snr_db,
                source_name=source_name,
                source_gain_db=source_gain_db,
                bg_gain_db=bg_gain_db,
                augment_source=augment_source,
                augment_bg=augment_bg,
            )

            # Write stems
            for name, wav in stems.items():
                save_audio(wav, track_dir / f"{name}.wav")

            if (idx + 1) % 100 == 0:
                print(f"  {split_name}: {idx + 1}/{n_tracks} tracks built")

        print(f"  {split_name}: {n_tracks}/{n_tracks} tracks built")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthesize instrument Demucs dataset")
    p.add_argument("--source-dir", required=True, type=Path, help="Isolated instrument clips")
    p.add_argument("--source-name", default="chinese-instrument", type=str,
                   help="Target instrument name for stem filenames (default: chinese-instrument)")
    p.add_argument("--bg-dir", required=True, type=Path, help="Instrument-free background music")
    p.add_argument("--out", default=Path("data/instrument_dataset"), type=Path)
    p.add_argument("--num-train", default=500, type=int)
    p.add_argument("--num-valid", default=50, type=int)
    p.add_argument("--seg-len", default=8.0, type=float, help="Segment length in seconds")
    p.add_argument("--snr-min", default=-5.0, type=float)
    p.add_argument("--snr-max", default=10.0, type=float)
    p.add_argument("--source-gain-db", default=-20.0, type=float)
    p.add_argument("--bg-gain-db", default=-20.0, type=float)
    p.add_argument("--no-augment", action="store_true",
                   help="Disable source augmentation (pitch/time/gain)")
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        source_dir=args.source_dir,
        bg_dir=args.bg_dir,
        out_dir=args.out,
        source_name=args.source_name,
        num_train=args.num_train,
        num_valid=args.num_valid,
        seg_len=args.seg_len,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        source_gain_db=args.source_gain_db,
        bg_gain_db=args.bg_gain_db,
        augment_source=not args.no_augment,
        seed=args.seed,
    )
