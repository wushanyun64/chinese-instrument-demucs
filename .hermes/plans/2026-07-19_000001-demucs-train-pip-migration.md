# Replace vendored demucs with pip-installed demucs[train]

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Replace `vendor/demucs/` with `pip install "demucs[train]"`. Upstream training docs explicitly support this — the vendored copy adds no value and causes constant PYTHONPATH friction.

**Architecture:** The `adefossez/demucs` PyPI package ships with `[train]` extras that include `dora-search`, `hydra-core`, and all training dependencies. Dora configs ship inside the wheel under `conf/`. The project's own `configs/` overlay (dset, variant) stays local. Training runs via `dora run` resolving base configs from the installed package + our overrides.

**Key insight from upstream training.md:**
> "Install the training dependencies with `pip install "demucs[train]"` (or, from a clone of the repository, `uv sync --extra train`)."

This is the documented, supported path. No source tree needed on disk.

**One open question (resolved in Task 2):** Does the demucs wheel include `conf/` for Dora config discovery, or do we point DORA_CONFIG_PATH at site-packages?

**Files affected (16 files):**
- Delete: `vendor/demucs/` (entire tree)
- Modify: `pyproject.toml`, `Makefile`, `README.md`
- Modify: `training/train.sh`, `training/patch_checkpoint.py`
- Modify: `inference/separate.py`, `eval/evaluate.py`
- Modify: `tests/conftest.py`, `tests/test_training_eval.py`
- Modify: `notebooks/01_quickstart.ipynb`
- Modify: `colab/chinese_instrument_demucs.ipynb`
- Modify: `docs/index.md`, `docs/installation.md`, `docs/inference.md`, `docs/evaluation.md`

---

### Task 1: Add demucs[train] to pyproject.toml

**Objective:** Replace vendored-era deps with the single upstream package.

**File:** `pyproject.toml`

**Change the dependencies block:**

Before:
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

After:
```toml
dependencies = [
    "torch>=2.0",
    "torchaudio>=2.0",
    "soundfile>=0.12",
    "numpy",
    "demucs[train]>=4.1.0",   # demucs + training stack (dora, hydra, musdb, museval, etc.)
    "julius",                  # used directly by data_pipeline/audio_utils.py
    "museval>=0.4",
]
```

Rationale for removals:
- `dora-search` → comes via `demucs[train]`
- `hydra-core` → comes via `demucs[train]`
- `omegaconf` → comes via `hydra-core`
- `openunmix` → may or may not be in upstream demucs (verify in Task 2); if present, it's transitive
- `einops` → comes via `demucs`

**Install:**

```bash
rm -rf .venv && uv sync --extra dev
```

**Verify basic import:**

```bash
uv run python -c "
import demucs; print('demucs:', demucs.__version__)
from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model
import dora; print('dora OK')
print('All core imports OK')
"
```

Expected: prints versions, no ModuleNotFoundError.

**Commit:**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: replace vendored demucs with demucs[train]>=4.1.0"
```

---

### Task 2: Verify Dora config discovery from installed package

**Objective:** Confirm that `dora run model=htdemucs` resolves configs from the pip-installed `demucs` package, and that all needed modules are accessible.

**Step 1: Find the installed conf directory**

```bash
uv run python -c "
from pathlib import Path
import demucs
pkg_dir = Path(demucs.__file__).parent
conf_dir = pkg_dir.parent / 'conf'
print('Package dir:', pkg_dir)
print('Conf dir:', conf_dir, 'exists:', conf_dir.is_dir())
if conf_dir.is_dir():
    for f in sorted(conf_dir.rglob('*.yaml'))[:10]:
        print(' ', f.relative_to(conf_dir))
"
```

Expected: `conf/` directory exists with `config.yaml` and subdirectories.

**Step 2: Verify dora can see the configs**

```bash
DORA_CONFIG_PATH="configs:$(uv run python -c "from pathlib import Path; import demucs; print(Path(demucs.__file__).parent.parent / 'conf')")" \
  uv run dora run --help 2>&1 | head -5
```

If dora is not found, ensure `demucs[train]` installed correctly:

```bash
uv run python -c "from dora import hydra_main; print('dora importable')"
```

**Step 3: Verify tools/ are accessible for model export**

```bash
uv run python -m tools.export --help 2>&1 || echo "tools/export not in wheel — needs source or alternative"
```

If `tools/export` is NOT in the pip wheel (likely — only `demucs/` is packaged), document the fallback: either copy the single `tools/export.py` script locally, or verify Dora checkpoints can be used directly with `-n SIG` in the separation CLI.

**Step 4: Verify openunmix availability**

```bash
uv run python -c "from openunmix.filtering import wiener; print('openunmix OK')"
```

If this fails, the upstream demucs may have replaced openunmix. Check what htdemucs.py imports:

```bash
uv run python -c "import demucs.htdemucs; print('htdemucs imports OK')"
```

If this fails, document the specific missing dependency and add it to pyproject.toml.

---

### Task 3: Delete vendor/demucs/

```bash
rm -rf vendor/
```

Verify vendor/ is gone or empty (if it contained only demucs/):

```bash
ls vendor/ 2>&1  # Expected: "No such file or directory"
```

```bash
git add vendor/  # stages the deletion
git commit -m "chore: remove vendored demucs/"
```

---

### Task 4: Update training/train.sh

**File:** `training/train.sh`

**Changes:**

1. Remove the PYTHONPATH export (lines 26-27):
   ```bash
   # REMOVE these lines:
   # Ensure vendored demucs is on PYTHONPATH
   export PYTHONPATH="${REPO_ROOT}/vendor/demucs:${PYTHONPATH:-}"
   ```

2. Update DORA_CONFIG_PATH to point at the installed package's conf directory:
   ```bash
   # Before:
   export DORA_CONFIG_PATH="${REPO_ROOT}/configs:${REPO_ROOT}/vendor/demucs/conf"
   
   # After:
   DEMUCS_CONF="$(uv run python -c "from pathlib import Path; import demucs; print(Path(demucs.__file__).parent.parent / 'conf')")"
   export DORA_CONFIG_PATH="${REPO_ROOT}/configs:${DEMUCS_CONF}"
   ```

3. Update the prerequisites comment:
   ```bash
   # Before:
   #   1. Dataset built:     uv run --env PYTHONPATH=vendor/demucs python data_pipeline/build_dataset.py ...
   #   2. Warm-start ready:  python training/patch_checkpoint.py
   
   # After:
   #   1. Dataset built:     uv run python data_pipeline/build_dataset.py ...
   #   2. Warm-start ready:  uv run python training/patch_checkpoint.py
   ```

4. Add a comment noting that `demucs[train]` must be installed (it is, via pyproject.toml):
   ```bash
   # Training stack is provided by demucs[train] in pyproject.toml — no extra setup needed.
   ```

**Verify (dry run):**

```bash
cd /home/jason_sun/Github/chinese-instrument-demucs && bash training/train.sh --help 2>&1 | head -10
```

**Commit:**

```bash
git add training/train.sh
git commit -m "refactor: update train.sh for pip-installed demucs[train]"
```

---

### Task 5: Clean up training/patch_checkpoint.py

**File:** `training/patch_checkpoint.py`

**Changes:**

Remove the vendored path injection (lines 19-24):

```python
# REMOVE:
# Ensure vendored demucs is importable
_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "demucs"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
```

Remove the `# noqa: E402` comments on the import lines (no longer needed since they're normal imports now):

```python
# Before:
from demucs.htdemucs import HTDemucs  # noqa: E402
from demucs.pretrained import get_model  # noqa: E402

# After:
from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model
```

Remove `import sys` if it's now unused.

**Verify:**

```bash
uv run python -c "from training.patch_checkpoint import build_model, load_and_patch; print('patch_checkpoint OK')"
```

**Commit:**

```bash
git add training/patch_checkpoint.py
git commit -m "refactor: remove vendor path hack from patch_checkpoint"
```

---

### Task 6: Clean up inference/separate.py and eval/evaluate.py

**Objective:** Remove `_VENDOR` path hacks.

**6a: `inference/separate.py`**

Remove line 16 (`_VENDOR = ...`) and change subprocess cwd:

```python
# REMOVE:
_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "demucs"

# CHANGE (line 61) from:
#   subprocess.run(cmd, check=True, cwd=_VENDOR.parent)
# to:
    subprocess.run(cmd, check=True)
```

Also remove unused `Path` import if `_VENDOR` was the only use.

**6b: `eval/evaluate.py`**

Remove lines 19-21:

```python
# REMOVE:
_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "demucs"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
```

Remove unused `sys` import.

**Verify:**

```bash
uv run python -c "from inference.separate import separate; print('separate OK')"
uv run python -c "from eval.evaluate import si_sdr, evaluate_dir; print('eval OK')"
```

**Commit:**

```bash
git add inference/separate.py eval/evaluate.py
git commit -m "refactor: remove vendor path hacks from inference and eval"
```

---

### Task 7: Clean up tests/

**7a: `tests/conftest.py`**

Replace entire file (its only purpose was the vendor path hack):

```python
"""Pytest configuration for chinese-instrument-demucs."""
# demucs is installed via pip — no special path setup needed.
```

Or delete the file if it now has no content worth keeping.

**7b: `tests/test_training_eval.py`**

Remove lines 7-15:

```python
# REMOVE:
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "demucs"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
```

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

### Task 8: Clean up Makefile

**File:** `Makefile`

Remove every occurrence of `--env PYTHONPATH=vendor/demucs` from all targets and comments.

After cleanup, targets should look like:

```makefile
env-verify:
	$(UV) run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
	$(UV) run python -c "import demucs; print('demucs OK')"

build-data:
	$(UV) run python data_pipeline/build_dataset.py
```

**Verify no stale refs:**

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

### Task 9: Clean up documentation

**Files:** `README.md`, `docs/index.md`, `docs/installation.md`, `docs/inference.md`, `docs/evaluation.md`

**Pattern to remove:** `--env PYTHONPATH=vendor/demucs` in all command examples.

**Also update `docs/installation.md`:** Remove or rewrite the "Vendored Demucs" section that explains vendoring. Replace with a note that demucs comes via pip:

```markdown
## Demucs

This project uses the [adefossez/demucs](https://github.com/adefossez/demucs) package installed via pip. Training, inference, and evaluation all work with the standard `pip install "demucs[train]"` — no vendored copy needed.
```

**Verify:**

```bash
grep -rn "vendor" README.md docs/ --include="*.md" 2>/dev/null
```

Expected: no output.

**Commit:**

```bash
git add README.md docs/
git commit -m "docs: remove vendored demucs references, document pip approach"
```

---

### Task 10: Clean up notebooks

**10a: `notebooks/01_quickstart.ipynb`**

Replace the sys.path walking cell (cell 1) with a simple import check:

```python
# Before (remove all this):
import os, sys
from pathlib import Path
REPO_ROOT = Path.cwd().resolve()
for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
    if (p / "vendor" / "demucs" / "demucs").is_dir():
        REPO_ROOT = p
        break
sys.path.insert(0, str(REPO_ROOT / "vendor" / "demucs"))

# After:
import torch
import demucs
print(f"PyTorch {torch.__version__}, Demucs {demucs.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**10b: `colab/chinese_instrument_demucs.ipynb`**

Colab-specific changes:
1. Add `!pip install "demucs[train]"` early in the notebook (replaces vendored setup cell)
2. Remove all `--env PYTHONPATH=vendor/demucs` from shell commands
3. Remove the "Set up vendored Demucs" section entirely
4. Update any sys.path manipulation cells

**Verify notebooks:**

```bash
uv run pytest --nbmake notebooks/ -v
```

**Commit:**

```bash
git add notebooks/ colab/
git commit -m "fix: remove vendor/demucs hacks from notebooks"
```

---

### Task 11: Final verification

**Step 1: Run full test suite**

```bash
uv run pytest tests/ -v
```

**Step 2: Verify all imports**

```bash
uv run python -c "
import demucs
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

**Step 3: Verify CLI works**

```bash
uv run python -m demucs --help 2>&1 | head -3
```

**Step 4: Verify no stale vendor references**

```bash
grep -rn "vendor/demucs" --include="*.py" --include="*.sh" --include="*.md" --include="*.ipynb" --include="Makefile" . 2>/dev/null | grep -v ".git/" | grep -v ".venv/" | grep -v "__pycache__" | grep -v ".hermes/"
```

Expected: no output.

**Step 5: Run notebook tests**

```bash
uv run pytest --nbmake notebooks/ -v
```

**Commit:**

```bash
git add -A
git commit -m "chore: final cleanup after vendor/demucs removal"
```

---

### Risks and Contingencies

1. **`conf/` not in pip wheel**: If the demucs wheel doesn't ship `conf/`, we keep a local copy of the base config YAML files (just `conf/config.yaml` and `conf/dset/` skeletons — ~5 files) and point DORA_CONFIG_PATH at them. Task 2 discovers this.

2. **`tools/export.py` not in pip wheel**: The `tools/` directory isn't packaged. If we need model export, copy `tools/export.py` (~30 lines) into our repo locally. Alternatively, check if Dora checkpoint signatures work directly with `demucs.separate.main -n SIG`.

3. **`openunmix` removed from upstream**: If upstream demucs dropped `openunmix`, add it as an explicit dependency.

4. **Dora resolves model=htdemucs from site-packages**: Dora discovers configs from `DORA_CONFIG_PATH`. If the installed package's conf isn't auto-discovered, we need to explicitly add the path (Task 2 handles this).
