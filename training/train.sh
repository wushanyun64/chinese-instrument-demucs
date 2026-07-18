#!/usr/bin/env bash
# Launch Demucs fine-tuning via Dora.
#
# Prerequisites:
#   1. Dataset built:     make build-data
#   2. Warm-start ready:  python training/patch_checkpoint.py
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
DSET="flute"
VARIANT="flute_ft"
# ------------------------------------------------------------------------

cd "$REPO_ROOT"

# Ensure vendored demucs is on PYTHONPATH
export PYTHONPATH="${REPO_ROOT}/vendor/demucs:${PYTHONPATH:-}"

# Point Dora at our config dirs
export DORA_CONFIG_PATH="${REPO_ROOT}/configs:${REPO_ROOT}/vendor/demucs/conf"

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
