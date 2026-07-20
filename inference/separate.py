"""CLI wrapper for separating a target instrument from audio using demucs.api.

Uses the upstream ``demucs.api.Separator`` class — no subprocess, no vendored paths.

Usage::

    python inference/separate.py input.wav --sig SIG --stem erhu
    python inference/separate.py input.wav --sig SIG --stem pipa
"""

from __future__ import annotations

import argparse
from pathlib import Path

import demucs.api


def separate(
    input_path: Path,
    sig: str,
    stem: str = "chinese-instrument",
    out_dir: Path | None = None,
    repo: Path | None = None,
    device: str = "cuda",
) -> Path:
    """Run Demucs separation and return the path to the target stem.

    Parameters
    ----------
    input_path : Path
        Audio file to process.
    sig : str
        Experiment signature of the trained model (e.g. ``"htdemucs"``
        or a Dora signature like ``"9357e12e"``).
    stem : str
        Target instrument name (default: ``"chinese-instrument"``).
    out_dir : Path | None
        Output directory (default: ``separated/<sig>/``).
    repo : Path | None
        Path to ``release_models/`` directory for locally trained models.
    device : str
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    Path to the extracted ``<stem>.wav``.
    """
    # Initialize separator — only pass repo if it exists (for locally trained models)
    kwargs: dict = {"model": sig, "device": device}
    if repo is not None and repo.exists():
        kwargs["repo"] = repo
    separator = demucs.api.Separator(**kwargs)

    # Separate
    origin, separated = separator.separate_audio_file(input_path)

    # Determine output paths
    if out_dir is None:
        out_dir = Path("separated") / sig
    out_dir.mkdir(parents=True, exist_ok=True)

    stem_name = input_path.stem
    track_dir = out_dir / stem_name
    track_dir.mkdir(parents=True, exist_ok=True)

    # Save all stems
    for source_name, source_wav in separated.items():
        out_file = track_dir / f"{source_name}.wav"
        demucs.api.save_audio(
            source_wav.cpu(),
            str(out_file),
            samplerate=separator.samplerate,
        )

    # Return the target stem path
    target_path = track_dir / f"{stem}.wav"
    if target_path.exists():
        return target_path

    # Fallback: search for the stem (handles naming variations)
    for p in track_dir.rglob(f"**/{stem}.wav"):
        return p

    raise FileNotFoundError(f"Target stem '{stem}' not found in {track_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Instrument stem separator")
    p.add_argument("input", type=Path, help="Input audio file")
    p.add_argument("--sig", required=True, help="Model experiment signature")
    p.add_argument(
        "--stem", default="chinese-instrument",
        help="Target instrument name (default: chinese-instrument)",
    )
    p.add_argument("--out", type=Path, default=None, help="Output directory")
    p.add_argument(
        "--repo", type=Path, default=Path("release_models"),
        help="Path to release_models/ (default: release_models/)",
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = separate(
        input_path=args.input,
        sig=args.sig,
        stem=args.stem,
        out_dir=args.out,
        repo=args.repo,
        device=args.device,
    )
    print(f"Done — stem at: {result}")
