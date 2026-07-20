#!/usr/bin/env bash
# Launch Demucs fine-tuning via Dora.
#
# Prerequisites:
#   1. Dataset built:   uv run python data_pipeline/build_dataset.py ...
#   2. Warm-start ready:  uv run python training/patch_checkpoint.py
#
# Usage:
#   bash training/train.sh
#   bash training/train.sh --clear   # fresh run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- Config ------------------------------------------------------------
DORA_ARGS="-d"                   # -d = all GPUs
MODEL="htdemucs"
DSET="instrument"
VARIANT="instrument_ft"
# ------------------------------------------------------------------------

cd "$REPO_ROOT"

# Demucs[train] is installed via pyproject.toml — no PYTHONPATH needed.
# Dora discovers base configs from our local copies (configs/demucs_base/)
# plus our project overrides (configs/).
export DORA_CONFIG_PATH="${REPO_ROOT}/configs:${REPO_ROOT}/configs/demucs_base"

echo "=== Launching training ==="
echo "  Model:   $MODEL"
echo "  Dataset: $DSET"
echo "  Variant: $VARIANT"
echo "  Args:    $DORA_ARGS $*"

exec dora run $DORA_ARGS \
    "model=$MODEL" \
    "dset=$DSET" \
    "variant=$VARIANT" \
    "$@"
