"""Non-strict warm-start loader: pretrained htdemucs (4 sources) → 2-source model.

Loads the pretrained ``htdemucs`` state dict, drops keys whose shapes don't
match our 2-source model (the output head projections), and saves a checkpoint
that Dora can resume from via ``continue_from``.

Usage::

    python training/patch_checkpoint.py --sources chinese-flute other \\
        --out outputs/warmstart.th
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure vendored demucs is importable
_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "demucs"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from demucs.htdemucs import HTDemucs  # noqa: E402
from demucs.pretrained import get_model  # noqa: E402


def build_model(sources: list[str], **kwargs) -> HTDemucs:
    """Build a fresh HTDemucs for our source list."""
    return HTDemucs(sources=sources, **kwargs)


def load_and_patch(
    sources: list[str],
    out_path: Path,
    pretrained_name: str = "htdemucs",
    device: str = "cpu",
) -> dict[str, object]:
    """Warm-start a 2-source model from a pretrained checkpoint.

    Returns a summary dict with ``loaded``, ``skipped``, and ``missing`` key counts.
    """
    print(f"Loading pretrained model: {pretrained_name}")
    pretrained = get_model(name=pretrained_name)
    pretrained_state = pretrained.state_dict()

    print(f"Building target model with sources: {sources}")
    target = build_model(sources=sources)
    target_state = target.state_dict()

    loaded: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []

    # Walk the target state dict and copy matching keys from pretrained
    for key, target_param in target_state.items():
        if key in pretrained_state:
            pretrained_param = pretrained_state[key]
            if pretrained_param.shape == target_param.shape:
                target_state[key] = pretrained_param
                loaded.append(key)
            else:
                skipped.append(
                    f"{key} (pretrained: {list(pretrained_param.shape)}, "
                    f"target: {list(target_param.shape)})"
                )
        else:
            missing.append(key)

    target.load_state_dict(target_state)

    # Save checkpoint
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": target_state}, out_path)

    # Log report
    print(f"\n--- Warm-start report ---")
    print(f"  Loaded: {len(loaded)} keys")
    print(f"  Skipped (shape mismatch): {len(skipped)} keys")
    for s in skipped:
        print(f"    - {s}")
    print(f"  Missing (new params): {len(missing)} keys")
    for m in missing:
        print(f"    - {m}")
    print(f"\nCheckpoint saved to: {out_path}")

    return {
        "loaded": len(loaded),
        "skipped": len(skipped),
        "missing": len(missing),
        "skipped_keys": skipped,
        "missing_keys": missing,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Non-strict warm-start: 4-source htdemucs → N-source model"
    )
    p.add_argument(
        "--sources", nargs="+", default=["chinese-flute", "other"],
        help="Target source list (default: chinese-flute other)",
    )
    p.add_argument(
        "--pretrained", default="htdemucs",
        help="Pretrained model name (default: htdemucs)",
    )
    p.add_argument(
        "--out", type=Path, default=Path("outputs/warmstart.th"),
        help="Output checkpoint path (default: outputs/warmstart.th)",
    )
    p.add_argument(
        "--device", default="cpu",
        help="Device to run on (default: cpu)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    load_and_patch(
        sources=args.sources,
        out_path=args.out,
        pretrained_name=args.pretrained,
        device=args.device,
    )
