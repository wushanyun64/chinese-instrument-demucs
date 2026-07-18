"""Evaluate SI-SDR / SDR metrics on a held-out real test set.

Usage::

    python eval/evaluate.py --test-dir data/real_test/ --sig <SIG>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "demucs"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))


def si_sdr(estimated: torch.Tensor, reference: torch.Tensor) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    Higher is better.  Typical range: −∞ to ~20 dB.
    """
    # Scale reference to match estimated
    alpha = (estimated * reference).sum() / (reference.square().sum() + 1e-10)
    reference_scaled = alpha * reference
    noise = estimated - reference_scaled
    ratio = reference_scaled.square().sum() / (noise.square().sum() + 1e-10)
    return float(10 * torch.log10(ratio + 1e-10))


def evaluate_dir(
    test_dir: Path,
    sig: str,
    repo: Path | None = None,
    device: str = "cuda",
) -> dict[str, object]:
    """Run separation on every file in *test_dir* and compute SI-SDR.

    Returns a dict with per-track metrics and a mean.
    """
    from inference.separate import separate

    test_files = sorted(
        p for p in test_dir.rglob("*")
        if p.suffix.lower() in {".wav", ".flac", ".mp3"}
    )

    results: dict[str, float] = {}
    for fp in test_files:
        # Separate
        est_path = separate(
            input_path=fp, sig=sig, repo=repo, device=device
        )
        # Load estimated and reference
        est_data, sr_est = sf.read(str(est_path))
        ref_path = fp.with_suffix(".stem.wav")  # convention: ground-truth instrument stem
        if not ref_path.exists():
            print(f"  Skipping {fp.name}: no ground-truth {ref_path.name}")
            continue
        ref_data, sr_ref = sf.read(str(ref_path))
        assert sr_est == sr_ref, f"Sample rate mismatch: {sr_est} vs {sr_ref}"

        est = torch.from_numpy(est_data.T).float()
        ref = torch.from_numpy(ref_data.T).float()

        # Ensure same length
        min_len = min(est.size(-1), ref.size(-1))
        est, ref = est[..., :min_len], ref[..., :min_len]

        score = si_sdr(est.mean(dim=0), ref.mean(dim=0))
        results[fp.name] = score
        print(f"  {fp.name}: SI-SDR = {score:.2f} dB")

    mean_score = float(np.mean(list(results.values()))) if results else 0.0
    print(f"\n  Mean SI-SDR: {mean_score:.2f} dB ({len(results)} tracks)")

    return {
        "per_track": results,
        "mean_si_sdr": mean_score,
        "sig": sig,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate instrument separation model")
    p.add_argument("--test-dir", type=Path, required=True,
                   help="Directory with real test mixtures + ground-truth .stem.wav stems")
    p.add_argument("--sig", required=True, help="Model experiment signature")
    p.add_argument("--repo", type=Path, default=Path("release_models"))
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out-json", type=Path, default=None,
                   help="Save metrics to JSON")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = evaluate_dir(
        test_dir=args.test_dir,
        sig=args.sig,
        repo=args.repo,
        device=args.device,
    )
    if args.out_json:
        args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Metrics saved to {args.out_json}")
