"""CLI wrapper for separating a target instrument from audio using the trained model.

Usage::

    python inference/separate.py input.wav --sig SIG --stem erhu
    python inference/separate.py input_folder/ --sig SIG --stem pipa
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "demucs"


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
        Audio file or directory to process.
    sig : str
        Experiment signature of the trained model.
    stem : str
        Target instrument name for ``--two-stems`` (default: ``"chinese-instrument"``).
    out_dir : Path | None
        Output directory (default: ``separated/<sig>/``).
    repo : Path | None
        Path to ``release_models/`` directory.
    device : str
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    Path to the extracted ``<stem>.wav`` (or the output directory for batch).
    """
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", stem,
        "-n", sig,
        "--device", device,
    ]
    if repo is not None:
        cmd.extend(["--repo", str(repo)])
    if out_dir is not None:
        cmd.extend(["--out", str(out_dir)])
    cmd.append(str(input_path))

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=_VENDOR.parent)

    # Determine output
    if out_dir is not None:
        result_dir = out_dir / sig
    else:
        result_dir = Path("separated") / sig

    if input_path.is_file():
        stem_name = input_path.stem
        stem_path = result_dir / stem_name / f"{stem}.wav"
        if stem_path.exists():
            return stem_path
        # Demucs may use different naming
        for p in result_dir.rglob(f"**/{stem}.wav"):
            return p

    return result_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Instrument stem separator")
    p.add_argument("input", type=Path, help="Input audio file or directory")
    p.add_argument("--sig", required=True, help="Model experiment signature")
    p.add_argument("--stem", default="chinese-instrument",
                   help="Target instrument name for --two-stems (default: chinese-instrument)")
    p.add_argument("--out", type=Path, default=None, help="Output directory")
    p.add_argument("--repo", type=Path, default=Path("release_models"),
                   help="Path to release_models/ (default: release_models/)")
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
