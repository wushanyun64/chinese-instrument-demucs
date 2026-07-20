# Replace vendored demucs with pip-installed demucs[train], use proper API

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Delete `vendor/demucs/`, install `demucs[train]` from PyPI, and refactor inference to use the programmatic `demucs.api.Separator` instead of subprocess. All paths (training, inference, eval) use the upstream pip package.

**Architecture:** Per upstream docs, `pip install "demucs[train]"` provides the full training stack (Dora, Hydra, configs). Inference uses `demucs.api.Separator` — a clean Python API that returns tensors, eliminating the subprocess hack. Training configs (dset, variant) live in our `configs/`; Dora discovers base configs from the installed package.

**Key upstream docs references:**
- API: `demucs.api.Separator(model=..., device=...).separate_audio_file(path)` → `(origin, {stem: tensor})`
- Training: `dora run -d model=htdemucs dset=... variant=...`
- Export: `python3 -m tools.export SIG` → `release_models/SIG.th`
- Separation CLI: `demucs --repo ./release_models -n SIG --two-stems <stem> file.wav`

**Files affected (18 files):**
- Delete: `vendor/demucs/` (entire tree)
- Modify: `pyproject.toml`, `Makefile`, `README.md`
- Modify: `training/train.sh`, `training/patch_checkpoint.py`
- Rewrite: `inference/separate.py` (subprocess → demucs.api.Separator)
- Modify: `eval/evaluate.py`
- Modify: `tests/conftest.py`, `tests/test_training_eval.py`
- Modify: `notebooks/01_quickstart.ipynb`
- Modify: `colab/chinese_instrument_demucs.ipynb`
- Modify: `docs/index.md`, `docs/installation.md`, `docs/inference.md`, `docs/evaluation.md`

**Verification:** `uv run pytest` passes, `python -c "import demucs; from demucs.api import Separator"` works.

---

### Task 1: Add demucs[train] to pyproject.toml, remove vendored-era deps

**File:** `pyproject.toml`

**Before:**
```toml
dependencies = [
    "torch>=2.0",
    "torchaudio>=2.0",
    "soundfile>=0.12",
    "numpy",
    # Demucs vendored dependencies
    "dora-search",
    "hydra-core",
    "omegaconf",
    "julius",
    "openunmix",
    "einops",
    # Evaluation / analysis
    "museval>=0.4",
]
```

**After:**
```toml
dependencies = [
    "torch>=2.0",
    "torchaudio>=2.0",
    "soundfile>=0.12",
    "numpy",
    "demucs[train]>=4.1.0",  # separation + training stack (dora, hydra, musdb, museval, etc.)
    "julius",                 # used directly by data_pipeline/audio_utils.py
    "museval>=0.4",
]
```

What's removed and why:
- `dora-search` → comes via `demucs[train]`
- `hydra-core` → via `demucs[train]` → `dora-search`
- `omegaconf` → via `hydra-core`
- `openunmix` → if still needed upstream, comes transitively; if not, dead code
- `einops` → comes via `demucs`

**Install:**
```bash
rm -rf .venv && uv sync --extra dev
```

**Verify:**
```bash
uv run python -c "
import demucs; print('demucs:', demucs.__version__)
from demucs.api import Separator; print('Separator OK')
from demucs.htdemucs import HTDemucs; print('HTDemucs OK')
from demucs.pretrained import get_model; print('get_model OK')
import dora; print('dora OK')
"
```

**Commit:**
```bash
git add pyproject.toml uv.lock
git commit -m "deps: replace vendored demucs with demucs[train]>=4.1.0"
```

---

### Task 2: Verify Dora config discovery and tools availability

**Step 1: Find installed conf directory**

```bash
uv run python -c "
from pathlib import Path
import demucs
conf_dir = Path(demucs.__file__).parent.parent / 'conf'
print('Conf dir:', conf_dir, 'exists:', conf_dir.is_dir())
if conf_dir.is_dir():
    for f in sorted(conf_dir.rglob('*.yaml'))[:12]:
        print(' ', f.relative_to(conf_dir))
else:
    print('  WARNING: conf/ not in wheel — will need local copy')
"
```

**Step 2: Check if tools/export is available**

```bash
uv run python -m tools.export --help 2>&1 || echo "NOT IN WHEEL — need local copy"
```

**Step 3: Check openunmix availability**

```bash
uv run python -c "from openunmix.filtering import wiener; print('openunmix OK')" 2>&1
```

If any of these fail, document the workaround (local conf copy, local export.py, add openunmix to deps).

**Step 4: Verify Dora can resolve model=htdemucs with our configs**

```bash
DEMUCS_CONF=$(uv run python -c "from pathlib import Path; import demucs; print(Path(demucs.__file__).parent.parent / 'conf')")
DORA_CONFIG_PATH="configs:${DEMUCS_CONF}" uv run dora run --help 2>&1 | head -5
```

Note the result — we'll use this DORA_CONFIG_PATH pattern in train.sh.

---

### Task 3: Delete vendor/demucs/

```bash
rm -rf vendor/
```

Verify:
```bash
ls vendor/ 2>&1  # Expected: "No such file or directory"
```

```bash
git add vendor/
git commit -m "chore: remove vendored demucs/"
```

---

### Task 4: Rewrite inference/separate.py to use demucs.api.Separator

**File:** `inference/separate.py`

**Current approach:** Subprocess to `python -m demucs --two-stems ...` — fragile, slow, depends on cwd.

**New approach:** Use the upstream `demucs.api.Separator` class — returns tensors directly, no subprocess, works cross-platform.

**Complete rewrite:**

```python
"""CLI wrapper for separating a target instrument from audio using demucs.api.

Usage::

    python inference/separate.py input.wav --sig SIG --stem erhu
    python inference/separate.py input_folder/ --sig SIG --stem pipa
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

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
        Experiment signature of the trained model.
    stem : str
        Target instrument name for ``--two-stems`` (default: ``"chinese-instrument"``).
    out_dir : Path | None
        Output directory (default: ``separated/<sig>/``).
    repo : Path | None
        Path to ``release_models/`` directory (used for locally trained models).
    device : str
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    Path to the extracted ``<stem>.wav``.
    """
    # Build model path if using local release_models/
    model_name = str(sig)
    if repo is not None and repo.exists():
        model_name = str(repo / f"{sig}.th")

    # Initialize separator with the trained model
    separator = demucs.api.Separator(
        model=model_name if (repo and repo.exists()) else sig,
        repo=repo,
        device=device,
    )

    # Separate
    origin, separated = separator.separate_audio_file(str(input_path))

    # Determine output path
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

    # Fallback: search for the stem
    for p in track_dir.rglob(f"**/{stem}.wav"):
        return p

    raise FileNotFoundError(f"Target stem '{stem}' not found in output")


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
```

**Key changes from current:**
1. No `subprocess` — uses `demucs.api.Separator` directly
2. No `_VENDOR` path hack — demucs is a normal pip import
3. No `sys` import needed
4. `separate_audio_file()` returns `(origin_tensor, {stem_name: stem_tensor})` — we save them with `demucs.api.save_audio()`
5. Handles both pip-installed models (`-n htdemucs`) and local trained models (`--repo ./release_models`)

**Update `eval/evaluate.py` to match:** The eval module calls `inference.separate.separate()` — update it to use the new API directly, or just let it work with the new signature (the return type is still `Path`).

**Verify:**
```bash
uv run python -c "from inference.separate import separate; print('import OK')"
```

If `sample_data/mixture_sample.wav` exists:
```bash
uv run python inference/separate.py sample_data/mixture_sample.wav --sig htdemucs --stem other --device cpu
```

Expected: creates `separated/htdemucs/mixture_sample/other.wav` with actual audio.

**Commit:**
```bash
git add inference/separate.py
git commit -m "refactor: use demucs.api.Separator instead of subprocess"
```

---

### Task 5: Update eval/evaluate.py

**File:** `eval/evaluate.py`

**Changes:**
1. Remove `_VENDOR` path injection (lines 19-21)
2. Remove unused `sys` import
3. The `separate()` call from `inference.separate` now returns a `Path` (same as before), so the core evaluation logic doesn't change — just the import path cleanup.

**Verify:**
```bash
uv run python -c "from eval.evaluate import si_sdr, evaluate_dir; print('eval OK')"
```

**Commit:**
```bash
git add eval/evaluate.py
git commit -m "refactor: remove vendor path hack from eval"
```

---

### Task 6: Update training/train.sh

**File:** `training/train.sh`

**Changes:**

1. Remove PYTHONPATH export (remove lines 26-27).
2. Update DORA_CONFIG_PATH to point at installed demucs conf:

```bash
# Before:
export DORA_CONFIG_PATH="${REPO_ROOT}/configs:${REPO_ROOT}/vendor/demucs/conf"

# After:
DEMUCS_CONF="$(uv run python -c "from pathlib import Path; import demucs; print(Path(demucs.__file__).parent.parent / 'conf')")"
export DORA_CONFIG_PATH="${REPO_ROOT}/configs:${DEMUCS_CONF}"
```

3. Update prerequisites comment.
4. Add note about demucs[train] already being in pyproject.toml.

**Verify (dry run):**
```bash
cd /home/jason_sun/Github/chinese-instrument-demucs && bash training/train.sh --help 2>&1 | head -10
```

**Commit:**
```bash
git add training/train.sh
git commit -m "refactor: use pip-installed demucs conf for Dora config"
```

---

### Task 7: Update training/patch_checkpoint.py

**File:** `training/patch_checkpoint.py`

**Changes:**
1. Remove lines 19-24 (`_VENDOR` path injection)
2. Remove `# noqa: E402` comments (imports are now normal)
3. Remove unused `sys` and `Path` imports (check if still needed)

**Verify:**
```bash
uv run python -c "from training.patch_checkpoint import build_model, load_and_patch; print('OK')"
```

**Commit:**
```bash
git add training/patch_checkpoint.py
git commit -m "refactor: remove vendor path hack from patch_checkpoint"
```

---

### Task 8: Clean up tests/

**8a: `tests/conftest.py`**

Replace entire file:
```python
"""Pytest configuration for chinese-instrument-demucs."""
# demucs is installed via pip — no special path setup needed.
```

**8b: `tests/test_training_eval.py`**

Remove lines 7-15 (VENDOR path injection + unused imports).

**Verify:**
```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

**Commit:**
```bash
git add tests/
git commit -m "refactor: remove vendor path hacks from tests"
```

---

### Task 9: Clean up Makefile

**File:** `Makefile`

Remove every `--env PYTHONPATH=vendor/demucs` from all targets and comments.

After cleanup, targets should look like:
```makefile
env-verify:
	$(UV) run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
	$(UV) run python -c "import demucs; print('demucs OK')"

build-data:
	$(UV) run python data_pipeline/build_dataset.py
```

**Verify:**
```bash
grep -n "PYTHONPATH\|vendor" Makefile
```
Expected: no output.

**Commit:**
```bash
git add Makefile
git commit -m "docs: remove PYTHONPATH=vendor/demucs from Makefile"
```

---

### Task 10: Clean up documentation

**Files:** `README.md`, `docs/index.md`, `docs/installation.md`, `docs/inference.md`, `docs/evaluation.md`

**Removals:**
- All `--env PYTHONPATH=vendor/demucs` from command examples
- The "Vendored Demucs" section in `docs/installation.md`

**Additions to `docs/installation.md`:**
```markdown
## Demucs

This project uses the official [adefossez/demucs](https://github.com/adefossez/demucs) package from PyPI.
Training, inference, and evaluation all work with `pip install "demucs[train]"` — no vendored copy needed.
```

**Update `docs/inference.md`:**
- Show `demucs.api.Separator` usage instead of subprocess
- Keep CLI one-liner as convenience wrapper

**Verify:**
```bash
grep -rn "vendor" README.md docs/ --include="*.md" 2>/dev/null
```
Expected: no output.

**Commit:**
```bash
git add README.md docs/
git commit -m "docs: remove vendored demucs, document pip + API approach"
```

---

### Task 11: Clean up notebooks

**11a: `notebooks/01_quickstart.ipynb`**

Replace the sys.path walking cell with:

```python
import torch
import demucs
print(f"PyTorch {torch.__version__}, Demucs {demucs.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Also update cell 3 to use `demucs.api.Separator` instead of subprocess:
```python
import demucs.api
from pathlib import Path

sample = Path('sample_data/mixture_sample.wav')
if sample.exists():
    separator = demucs.api.Separator(model='htdemucs', device='cuda' if torch.cuda.is_available() else 'cpu')
    origin, separated = separator.separate_audio_file(str(sample))
    print(f'Separated stems: {list(separated.keys())}')
else:
    print('Add sample_data/mixture_sample.wav')
```

**11b: `colab/chinese_instrument_demucs.ipynb`**

1. Replace vendored setup cell with `!pip install "demucs[train]"`
2. Remove all `--env PYTHONPATH=vendor/demucs` from shell commands
3. Remove sys.path manipulation cells
4. Update inference cells to use `demucs.api.Separator`

**Verify notebooks:**
```bash
uv run pytest --nbmake notebooks/ -v
```

**Commit:**
```bash
git add notebooks/ colab/
git commit -m "fix: remove vendor/demucs hacks, use demucs.api in notebooks"
```

---

### Task 12: Final verification

**Step 1: Full test suite**
```bash
uv run pytest tests/ -v
```

**Step 2: All imports**
```bash
uv run python -c "
import demucs
from demucs.api import Separator
from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model
from data_pipeline.audio_utils import load_audio, save_audio, mix_at_snr
from data_pipeline.build_dataset import build_track, build_dataset
from training.patch_checkpoint import build_model, load_and_patch
from inference.separate import separate
from eval.evaluate import si_sdr, evaluate_dir
print('All imports OK')
"
```

**Step 3: Separation smoke test**
```bash
uv run python inference/separate.py sample_data/mixture_sample.wav --sig htdemucs --stem other --device cpu
```
Expected: creates separated stems, no errors.

**Step 4: No stale vendor refs**
```bash
grep -rn "vendor/demucs" --include="*.py" --include="*.sh" --include="*.md" --include="*.ipynb" --include="Makefile" . 2>/dev/null | grep -v ".git/" | grep -v ".venv/" | grep -v "__pycache__" | grep -v ".hermes/"
```
Expected: no output.

**Step 5: Notebook tests**
```bash
uv run pytest --nbmake notebooks/ -v
```

**Commit (any stragglers):**
```bash
git add -A
git commit -m "chore: final verification after vendor removal"
```

---

### Risk / Contingency Table

| Risk | Likelihood | Mitigation (checked in Task 2) |
|------|-----------|-------------------------------|
| `conf/` not in pip wheel | Low (training.md references it) | Keep local copy of base config YAMLs (~5 files) |
| `tools/export.py` not in wheel | High (`tools/` not in hatch packages list) | Copy single script locally, or verify Dora checkpoints work directly with `-n SIG` |
| `openunmix` removed from upstream | Medium | Add as explicit dep in pyproject.toml |
| Dora can't find conf in site-packages | Low (Hydra config discovery is path-based) | Explicitly add path via DORA_CONFIG_PATH (already in train.sh) |
| `Separator` model path format differs from `subprocess` approach | Medium | Test with local trained model in Task 4 verification step |
