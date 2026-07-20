# Migrate from vendored demucs to pip-installed demucs

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Replace the vendored `vendor/demucs/` directory with a proper `pip install demucs` dependency, removing all PYTHONPATH hacks across the codebase.

**Architecture:** Demucs is a maintained PyPI package (v4.1.0, `adefossez/demucs`). The vendored copy is an outdated facebookresearch-era snapshot that causes real friction: every command needs `PYTHONPATH=vendor/demucs`, every notebook needs `sys.path` manipulation, and Jupyter kernels can't find demucs. The pip-installed package exposes the same API (`HTDemucs`, `get_model`, `python -m demucs`) and brings its own Dora/Hydra configs.

**Tech Stack:** uv + pip, demucs>=4.1.0, upstream Dora/Hydra config discovery

**Risk:** Upstream demucs v4.1.0 may have diverged from the vendored copy. Specifically, `openunmix` (imported by `htdemucs.py` as `from openunmix.filtering import wiener`) may no longer be a dependency — this needs verification in Task 2.

**Files affected (16 files):**
- Delete: `vendor/demucs/` (entire tree, ~30+ .py files + __pycache__)
- Modify: `pyproject.toml`, `Makefile`, `README.md`
- Modify: `training/train.sh`, `training/patch_checkpoint.py`
- Modify: `inference/separate.py`, `eval/evaluate.py`
- Modify: `tests/conftest.py`, `tests/test_training_eval.py`
- Modify: `notebooks/01_quickstart.ipynb`
- Modify: `colab/chinese_instrument_demucs.ipynb`
- Modify: `docs/index.md`, `docs/installation.md`, `docs/inference.md`, `docs/evaluation.md`

**Verification:** `uv run pytest` passes, `python -c "import demucs; from demucs.htdemucs import HTDemucs"` works without PYTHONPATH.

---

### Task 1: Add demucs to pyproject.toml and install

**Objective:** Declare the pip dependency and install it.

**Files:**
- Modify: `pyproject.toml` (dependencies section)

**Step 1: Add `demucs>=4.1.0` to dependencies**

In `pyproject.toml`, add `"demucs>=4.1.0"` to the `dependencies` list. Also consider whether any vendored-era deps can now be dropped — they come transitively via demucs:

Current vendored-era deps to audit:
- `dora-search` — used by Dora directly for training. Keep.
- `hydra-core` — used by Dora. Keep.
- `omegaconf` — used by Hydra. Keep.
- `openunmix` — only imported inside vendored demucs. **Remove** — will come transitively if needed, or may be dead.
- `julius` — used by `data_pipeline/audio_utils.py` directly. Keep.
- `einops` — will come transitively via demucs. Keep or let it be transitive.

Exact change to `pyproject.toml`:

```diff
 dependencies = [
     "torch>=2.0",
     "torchaudio>=2.0",
     "soundfile>=0.12",
     "numpy",
+    "demucs>=4.1.0",
     # Demucs vendored dependencies
     "dora-search",
     "hydra-core",
     "omegaconf",
-    "julius",
-    "openunmix",
-    "einops",
+    "julius",    # used directly by data_pipeline/audio_utils.py
     # Evaluation / analysis
     "museval>=0.4",
 ]
```

Wait — `einops` is also used directly? Let's check... No, it's only in vendored demucs. Remove it.

Also — `demucs` on PyPI depends on `torch>=2.1`. Our `torch>=2.0` is fine, they're compatible.

**Step 2: Install**

```bash
uv sync --extra dev
```

Expected: demucs v4.1.0 installed, no errors.

**Step 3: Quick import check**

```bash
uv run python -c "import demucs; from demucs.htdemucs import HTDemucs; from demucs.pretrained import get_model; print('demucs version:', demucs.__version__); print('All imports OK')"
```

Expected: prints version and "All imports OK". No PYTHONPATH needed.

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add demucs>=4.1.0, remove vendored-transitive deps"
```

---

### Task 2: Verify upstream demucs API compatibility

**Objective:** Confirm that the pip-installed demucs exposes everything this project needs. Catch any API drift before proceeding.

**Verify these imports work:**

```bash
uv run python -c "
from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model
from demucs.train import main as train_main  # needed by Dora
print('HTDemucs sources param:', HTDemucs.__init__.__code__.co_varnames)
"
```

Expected: no errors. The `sources` parameter should be in `HTDemucs.__init__` signature.

**Verify `python -m demucs` works:**

```bash
uv run python -m demucs --help
```

Expected: prints usage, `--two-stems` flag exists.

**Verify Dora config directory exists in installed package:**

```bash
python -c "import demucs; from pathlib import Path; p = Path(demucs.__file__).parent.parent / 'conf'; print('conf dir:', p, 'exists:', p.is_dir())"
```

This checks whether the upstream wheel ships the `conf/` directory (needed by Dora for training). If it doesn't, we need to keep a local copy of the training configs.

Expected: `conf/` directory exists inside the installed demucs package, OR we discover it doesn't and need to adjust the plan.

**If conf/ is missing:** Create a local `configs/demucs_base/` with copied configs from the upstream repo, and update `DORA_CONFIG_PATH` to include it.

---

### Task 3: Remove vendor/demucs/ directory

**Objective:** Delete the vendored code.

```bash
rm -rf vendor/
git add vendor/
```

Note: if `vendor/` contained anything else besides `demucs/`, only remove `vendor/demucs/`.

**Step 1: Verify it's gone**

```bash
ls vendor/ 2>&1
```

Expected: "No such file or directory".

**Step 2: Commit**

```bash
git commit -m "chore: remove vendored demucs/"
```

---

### Task 4: Clean up training/ scripts

**Objective:** Remove all references to `vendor/demucs` from training modules.

**Files:**
- Modify: `training/patch_checkpoint.py`
- Modify: `training/train.sh`

**4a: `training/patch_checkpoint.py`**

Remove these lines (21-24):

```python
# Ensure vendored demucs is importable
_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "demucs"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
```

And remove the `import sys` and `Path` imports if they're now unused. Also remove the `# noqa: E402` comments on lines 26-27 since the imports are now normal.

```bash
uv run python -c "from training.patch_checkpoint import build_model, load_and_patch; print('patch_checkpoint imports OK')"
```

**4b: `training/train.sh`**

Remove lines 26-27 (PYTHONPATH for vendor):

```bash
# Remove these lines:
export PYTHONPATH="${REPO_ROOT}/vendor/demucs:${PYTHONPATH:-}"
```

Update line 30 — change DORA_CONFIG_PATH to point at the installed demucs configs:

```bash
# Before:
export DORA_CONFIG_PATH="${REPO_ROOT}/configs:${REPO_ROOT}/vendor/demucs/conf"

# After:
export DORA_CONFIG_PATH="${REPO_ROOT}/configs"
```

The installed demucs package's conf should be auto-discovered by Dora since demucs is installed and Dora looks in the package's conf directory. If not, add the installed path explicitly (discovered in Task 2).

Also update the prerequisites comment (lines 4-5):

```bash
# Before:
#   1. Dataset built:     uv run --env PYTHONPATH=vendor/demucs python data_pipeline/build_dataset.py ...

# After:
#   1. Dataset built:     uv run python data_pipeline/build_dataset.py ...
```

**Step: Verify**

```bash
bash training/train.sh --help 2>&1 | head -5
```

Expected: Dora help output (or "command not found" for dora if not on PATH — that's fine, the script is meant to be run with dora installed).

**Commit:**

```bash
git add training/
git commit -m "refactor: remove vendor/demucs refs from training scripts"
```

---

### Task 5: Clean up inference/ and eval/

**Objective:** Remove `_VENDOR` path hacks from inference and evaluation modules.

**Files:**
- Modify: `inference/separate.py`
- Modify: `eval/evaluate.py`

**5a: `inference/separate.py`**

Remove line 16 (`_VENDOR = ...`) and update the subprocess `cwd` on line 61. Currently `separate()` runs `subprocess.run(cmd, check=True, cwd=_VENDOR.parent)`. Change `cwd` to `None` (or the repo root) since `python -m demucs` is now an installed entry point:

```python
# Remove line 16:
# _VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "demucs"

# Change line 61 from:
#   subprocess.run(cmd, check=True, cwd=_VENDOR.parent)
# to:
    subprocess.run(cmd, check=True)
```

Also remove unused `Path` import if it's only used for `_VENDOR`.

**5b: `eval/evaluate.py`**

Remove lines 19-21 (the `_VENDOR` path injection):

```python
# Remove:
_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "demucs"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
```

Also remove unused `sys` import.

**Step: Verify**

```bash
uv run python -c "from inference.separate import separate; print('separate imports OK')"
uv run python -c "from eval.evaluate import si_sdr, evaluate_dir; print('eval imports OK')"
```

**Commit:**

```bash
git add inference/ eval/
git commit -m "refactor: remove vendor/demucs refs from inference and eval"
```

---

### Task 6: Clean up tests/

**Objective:** Remove vendored path hacks from test files.

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_training_eval.py`

**6a: `tests/conftest.py`**

Replace the entire file content — the existing file's only purpose was adding vendor/demucs to sys.path:

```python
"""Pytest configuration for chinese-instrument-demucs."""
# No special path setup needed — demucs is now a pip dependency.
```

Or just delete the file entirely if nothing else is in it.

**6b: `tests/test_training_eval.py`**

Remove lines 7-15 (the VENDOR path injection):

```python
# Remove:
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "demucs"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
```

**Step: Verify tests pass**

```bash
uv run pytest tests/ -v
```

Expected: All tests pass without any PYTHONPATH manipulation.

**Commit:**

```bash
git add tests/
git commit -m "refactor: remove vendor/demucs refs from tests"
```

---

### Task 7: Clean up Makefile

**Objective:** Remove all `--env PYTHONPATH=vendor/demucs` from Makefile targets.

**Files:**
- Modify: `Makefile`

Replace every occurrence of `--env PYTHONPATH=vendor/demucs ` with nothing. The targets affected:

| Line | Target | Change |
|------|--------|--------|
| 7-10 | comments | Remove `--env PYTHONPATH=vendor/demucs` |
| 16 | `env` | Remove from echo message |
| 19-20 | `env-verify` | Remove from both commands |
| 23 | `build-data` | Remove |
| 26 | `validate-data` | Remove |
| 32 | `separate` | Remove |
| 35 | `eval` | Remove |
| 38 | `test` | Remove |
| 41 | `notebooks` | Remove |

After cleanup, each target should look like:

```makefile
env-verify:
	$(UV) run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
	$(UV) run python -c "import demucs; print('demucs OK')"
```

**Step: Verify Makefile is clean**

```bash
grep -n "vendor" Makefile
```

Expected: no output.

**Commit:**

```bash
git add Makefile
git commit -m "docs: remove PYTHONPATH=vendor/demucs from Makefile"
```

---

### Task 8: Clean up documentation

**Objective:** Update README.md and docs/ to remove all vendor references.

**Files:**
- Modify: `README.md`
- Modify: `docs/index.md`
- Modify: `docs/installation.md`
- Modify: `docs/inference.md`
- Modify: `docs/evaluation.md`

**Pattern to remove everywhere:** `--env PYTHONPATH=vendor/demucs` in command examples.

And in `docs/installation.md`, remove or rewrite the "Vendored Demucs" section (around line 33) that explains why the repo vendors demucs.

Example fix for `README.md`:

```diff
- uv run --env PYTHONPATH=vendor/demucs python data_pipeline/build_dataset.py \
+ uv run python data_pipeline/build_dataset.py \
```

**Step: Verify no vendor refs remain in docs**

```bash
grep -rn "vendor" README.md docs/ 2>/dev/null
```

Expected: no output (except possibly a line about "vendor" in the context of something else — use judgment).

**Commit:**

```bash
git add README.md docs/
git commit -m "docs: remove all vendor/demucs references"
```

---

### Task 9: Clean up notebooks

**Objective:** Remove sys.path hacks from notebooks and update to use pip-installed demucs.

**Files:**
- Modify: `notebooks/01_quickstart.ipynb`
- Modify: `colab/chinese_instrument_demucs.ipynb`

**9a: `notebooks/01_quickstart.ipynb`**

Replace the path-finding cell (cell 1) with a simple import check:

```python
# Old cell content (remove all the REPO_ROOT walking code):
import os, sys
from pathlib import Path
REPO_ROOT = Path.cwd().resolve()
for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
    if (p / "vendor" / "demucs" / "demucs").is_dir():
        REPO_ROOT = p
        break
sys.path.insert(0, str(REPO_ROOT / "vendor" / "demucs"))

# Replace with:
import torch
print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**9b: `colab/chinese_instrument_demucs.ipynb`**

The Colab notebook needs different treatment since it doesn't have a local venv pre-installed. Key changes:

1. Remove cell 7 ("Set up vendored Demucs") and cell 8 (sys.path.insert) — or replace them with `!pip install demucs`.
2. In cells that call `!uv run --env PYTHONPATH=vendor/demucs ...`, remove the `--env PYTHONPATH=vendor/demucs` part.
3. If the Colab uses `!pip install demucs` early, demucs will be importable normally.

**Step: Verify notebooks**

```bash
uv run pytest --nbmake notebooks/ -v
```

**Commit:**

```bash
git add notebooks/ colab/
git commit -m "fix: remove vendor/demucs hacks from notebooks"
```

---

### Task 10: Final verification

**Objective:** Run the full test suite and import checks to confirm nothing is broken.

**Step 1: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all pass.

**Step 2: Verify all key imports work cleanly**

```bash
uv run python -c "
# Core demucs
import demucs
from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model

# Project modules
from data_pipeline.audio_utils import load_audio, save_audio, mix_at_snr
from data_pipeline.build_dataset import build_track, build_dataset
from data_pipeline.validate_contamination import validate
from training.patch_checkpoint import build_model, load_and_patch
from inference.separate import separate
from eval.evaluate import si_sdr, evaluate_dir

print('All imports OK')
"
```

Expected: "All imports OK".

**Step 3: Verify `python -m demucs` CLI works**

```bash
uv run python -m demucs --help 2>&1 | head -3
```

Expected: demucs help text.

**Step 4: Verify no stale vendor references remain**

```bash
grep -rn "vendor/demucs" --include="*.py" --include="*.sh" --include="*.md" --include="*.ipynb" --include="Makefile" . 2>/dev/null | grep -v ".git/" | grep -v ".venv/" | grep -v "__pycache__"
```

Expected: no output.

**Step 5: Run notebook tests**

```bash
uv run pytest --nbmake notebooks/ -v
```

**Commit:**

```bash
git add -A
git commit -m "chore: final verification after vendor/demucs removal"
```

---

### Open Questions / Risks

1. **Upstream `conf/` directory**: Does the PyPI demucs wheel include the `conf/` directory needed by Dora? If not, we need to keep a local copy of the base Dora configs. (Checked in Task 2.)

2. **`openunmix` dependency**: The vendored demucs imports `from openunmix.filtering import wiener`. Upstream demucs v4.1.0 may have removed or replaced this. (Checked in Task 2.)

3. **Dora model discovery**: Dora resolves `model=htdemucs` by searching config paths. With demucs pip-installed, Dora should find its configs automatically, but if not, we may need to add the site-packages path to `DORA_CONFIG_PATH`.

4. **Colab notebook**: The Colab notebook installs deps differently. It may need `!pip install demucs` added explicitly since `uv sync --extra dev` doesn't apply to Colab's runtime.
