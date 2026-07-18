# chinese-flute-demucs → chinese-instrument-demucs Migration Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Generalize the single-target Demucs separator from hardcoded "Chinese flute" to any Chinese instrument, making the target instrument name configurable.

**Architecture:** Two-source model `['<instrument>', 'other']` where `other = mixture − instrument`. The instrument name is a CLI/config parameter; all file naming, directory structures, and log output follow. Keep the single-target design, synthetic data pipeline, and warm-start loader — only broaden from flute-specific to instrument-agnostic.

**Tech Stack:** Python 3.10+, torch, demucs (vendored), Dora/Hydra, soundfile, uv.

---

## Key design decision (settled — do not re-litigate)

- **Single-target with configurable instrument name.** The source list is `['<instrument>', 'other']`. The instrument name passes through CLI args (`--source-name erhu`, defaults to `chinese-instrument`). This keeps the 2-source Demucs invariant intact while supporting any instrument.
- **No multi-instrument support.** Each trained model still targets ONE instrument. If a user wants erhu AND pipa separators, they train two models with different `--source-name` values.
- **Backward-incompatible rename.** Filenames, config keys, CLI flags, and output directories all change. Old flute-trained checkpoints won't load without renaming the source key. Document this in the migration guide.

---

## Phase 0: Repo rename

### Task 1: Rename repo directory and package name

**Objective:** Rename `chinese-flute-demucs` to `chinese-instrument-demucs`.

**Files:**
- Rename: `~/Github/chinese-flute-demucs/` → `~/Github/chinese-instrument-demucs/`
- Modify: `pyproject.toml:6,8`

**Step 1: Rename directory**

```bash
mv ~/Github/chinese-flute-demucs ~/Github/chinese-instrument-demucs
```

**Step 2: Update pyproject.toml**

```toml
name = "chinese-instrument-demucs"
description = "Single-target Demucs source separator for Chinese instruments"
```

**Step 3: Commit**

```bash
git add -A
git commit -m "refactor: rename repo to chinese-instrument-demucs"
```

---

## Phase 1: Core code — parameterize the instrument name

### Task 2: Refactor audio_utils.py — no source-specific strings

**Objective:** Verify `audio_utils.py` has no instrument-specific strings. It already doesn't — loudness, SNR, resampling are instrument-agnostic. No changes needed; this task is a verification pass.

**Files:**
- Verify: `data_pipeline/audio_utils.py`

**Step 1: Confirm no 'flute' references**

```bash
grep -i flute data_pipeline/audio_utils.py  # should return nothing
```

### Task 3: Refactor build_dataset.py — configurable source name

**Objective:** Replace hardcoded `'chinese-flute'` with a CLI parameter `--source-name` (default: `'chinese-instrument'`). Rename `--flute-dir` → `--source-dir`.

**Files:**
- Modify: `data_pipeline/build_dataset.py`
- Test: `tests/test_data_pipeline.py` (update test file references)

**Step 1: Update module docstring and CLI args**

Replace:
- `--flute-dir` → `--source-dir`
- `--flute-gain-db` → `--source-gain-db`
- All `'chinese-flute'` string literals → `args.source_name`
- All `'flute'` variable names → `'source'` (internal variables only; public API preserved)
- Output stem filename: `f"{source_name}.wav"` instead of `chinese-flute.wav`
- `flute_dataset/` default output → `instrument_dataset/`

**Step 2: Update build_track() signature**

```python
def build_track(
    source_file: Path,      # was flute_file
    bg_file: Path,
    seg_samples: int,
    snr_db: float,
    source_name: str = "chinese-instrument",  # NEW
    augment_source: bool = True,  # was augment_flute
    augment_bg: bool = False,
    source_gain_db: float = -20.0,  # was flute_gain_db
    bg_gain_db: float = -20.0,
) -> dict[str, torch.Tensor]:
```

Returns `{source_name: ..., 'other': ..., 'mixture': ...}` (was `{'chinese-flute': ..., ...}`).

**Step 3: Update build_dataset() signature**

Replace `flute_dir` → `source_dir`, `flute_gain_db` → `source_gain_db`. Internal `flute_files` → `source_files`, `train_flutes` → `train_sources`.

**Step 4: Run tests to verify failure**

```bash
make test
```
Expected: FAIL — test assertions reference `'chinese-flute'` which no longer exists.

**Step 5: Update tests**

In `tests/test_data_pipeline.py`:
- All `'chinese-flute'` string references → `'chinese-instrument'` (default test source name)
- All `flute_file` → `source_file`, `flute_dir` → `source_dir`
- Test fixture directories: `flute_*.wav` → `source_*.wav`
- Update `TestBuildTrack` docstrings

**Step 6: Run tests to verify pass**

```bash
make test
```
Expected: 32 passed.

**Step 7: Commit**

```bash
git add data_pipeline/build_dataset.py tests/test_data_pipeline.py
git commit -m "refactor: parameterize instrument name in build_dataset (--source-name)"
```

### Task 4: Refactor validate_flute_free.py → validate_contamination.py

**Objective:** Rename and generalize the background contamination checker. Keep the spectral heuristic (it's already frequency-band-based, not flute-specific) but rename files/variables/CLI.

**Files:**
- Rename: `data_pipeline/validate_flute_free.py` → `data_pipeline/validate_contamination.py`
- Modify: `tests/test_data_pipeline.py` (update imports)

**Step 1: Rename file and update docstring**

- `FLUTE_BAND_LO/HI` → keep as-is (they're generic frequency bands, not flute-specific; add comment explaining they target the typical vocal/instrument midrange)
- Function docstrings: "flute" → "target instrument"
- CLI description: "flute contamination" → "target-instrument contamination"
- Internal function `_flute_band_ratio` → `_instrument_band_ratio`
- Module docstring: update to generic language

**Step 2: Update test imports**

In `tests/test_data_pipeline.py`:
```python
from data_pipeline.validate_contamination import _instrument_band_ratio, validate
```

And update `TestFluteBandRatio` → `TestInstrumentBandRatio`.

**Step 3: Run tests**

```bash
make test
```
Expected: 32 passed.

**Step 4: Commit**

```bash
git add data_pipeline/validate_contamination.py tests/test_data_pipeline.py
git rm data_pipeline/validate_flute_free.py
git commit -m "refactor: generalize validate_flute_free → validate_contamination"
```

### Task 5: Refactor inference/separate.py — configurable stem name

**Objective:** Replace hardcoded `'chinese-flute'` with parameter.

**Files:**
- Modify: `inference/separate.py`
- Test: `tests/test_training_eval.py`

**Step 1: Add `--stem` parameter**

Add `--stem` argument (default `'chinese-instrument'`). This controls:
- The `--two-stems` argument passed to demucs CLI
- The output filename pattern searched for

**Step 2: Update function signature**

```python
def separate(
    input_path: Path,
    sig: str,
    stem: str = "chinese-instrument",  # NEW
    ...
) -> Path:
```

Replace all `'chinese-flute'` string literals with `stem`.

**Step 3: Run tests — verify test_eval still works**

The eval tests don't call `separate()` directly in tests (just `si_sdr`). The `TestSISDR` tests should pass unchanged.

**Step 4: Commit**

```bash
git add inference/separate.py
git commit -m "refactor: make stem name configurable in inference/separate.py"
```

### Task 6: Refactor eval/evaluate.py — configurable ground-truth suffix

**Objective:** Replace `.flute.wav` ground-truth convention with `.stem.wav`.

**Files:**
- Modify: `eval/evaluate.py`

**Step 1: Update ground-truth file convention**

Change `fp.with_suffix(".flute.wav")` → `fp.with_suffix(".stem.wav")`.

**Step 2: Update docstrings and CLI descriptions**

"flute" → "target instrument" throughout.

**Step 3: Commit**

```bash
git add eval/evaluate.py
git commit -m "refactor: generalize eval ground-truth suffix .flute.wav → .stem.wav"
```

### Task 7: Refactor training/patch_checkpoint.py — CLI default update

**Objective:** Update the default `--sources` in the CLI to be generic. The loader itself is already instrument-agnostic (takes any source list).

**Files:**
- Modify: `training/patch_checkpoint.py:9-10`

**Step 1: Update default and docstring**

Default `--sources` from `['chinese-flute', 'other']` → `['chinese-instrument', 'other']`.

**Step 2: Run tests to verify**

Tests build 2-source vs 4-source models with explicit source lists — they don't depend on the CLI default. Should pass unchanged.

**Step 3: Commit**

```bash
git add training/patch_checkpoint.py
git commit -m "refactor: update default sources in patch_checkpoint CLI"
```

---

## Phase 2: Configs and orchestration

### Task 8: Rename config files

**Objective:** Rename `flute.yaml` → `instrument.yaml`, `flute_ft.yaml` → `instrument_ft.yaml`.

**Files:**
- Rename: `configs/dset/flute.yaml` → `configs/dset/instrument.yaml`
- Rename: `configs/variant/flute_ft.yaml` → `configs/variant/instrument_ft.yaml`
- Modify: `training/train.sh:20-21`

**Step 1: Rename and update content**

`configs/dset/instrument.yaml`:
```yaml
# Dataset config for Chinese instrument training
dset:
  wav: /abs/path/to/data/instrument_dataset
  sources: ['chinese-instrument', 'other']
```

`configs/variant/instrument_ft.yaml`:
```yaml
# Fine-tune variant for Chinese instrument separator
```

**Step 2: Update train.sh**

```bash
DSET="instrument"
VARIANT="instrument_ft"
```

**Step 3: Commit**

```bash
git add configs/ training/train.sh
git rm configs/dset/flute.yaml configs/variant/flute_ft.yaml
git commit -m "refactor: rename configs flute → instrument"
```

---

## Phase 3: Documentation

### Task 9: Update all docs — batch replace

**Objective:** Replace "Chinese flute" → "Chinese instrument", "flute" → "target instrument" contextually across all docs, keeping technical accuracy.

**Files:**
- `README.md`
- `docs/index.md`
- `docs/installation.md`
- `docs/concepts.md`
- `docs/data.md`
- `docs/training.md`
- `docs/inference.md`
- `docs/evaluation.md`
- `docs/troubleshooting.md`
- `docs/faq.md`
- `mkdocs.yml`
- `sample_data/README.md`

**Step 1: Batch edits per file**

For each file:
- Title/heading references: "Chinese Flute" → "Chinese Instrument"
- Concept explanations: describe the general single-target design, using flute as *example* not identity
- CLI examples: update `--flute-dir` → `--source-dir`, `--flute-gain-db` → `--source-gain-db`, `--two-stems chinese-flute` → `--two-stems <instrument>`
- Config references: `dset=flute` → `dset=instrument`, `variant=flute_ft` → `variant=instrument_ft`
- File paths: `flute_dataset/` → `instrument_dataset/`, `flute_clips/` → `source_clips/`
- Source names in code blocks: `'chinese-flute'` → `'chinese-instrument'`
- Troubleshooting: update symptom descriptions to be instrument-agnostic
- FAQ: "Can it separate other instruments?" → update answer reflecting the parameterized design

**Step 2: Commit**

```bash
git add docs/ README.md mkdocs.yml sample_data/README.md
git commit -m "docs: generalize all documentation from flute to instrument"
```

### Task 10: Update notebooks

**Objective:** Update all 5 notebooks for instrument-agnostic naming.

**Files:**
- `notebooks/01_quickstart.ipynb`
- `notebooks/02_build_dataset.ipynb`
- `notebooks/03_train_and_finetune.ipynb`
- `notebooks/04_evaluate_and_listen.ipynb`
- `notebooks/05_batch_inference.ipynb`

**Step 1: Update each notebook**

- Code cells: `FLUTE_DIR` → `SOURCE_DIR`, `'chinese-flute'` → `'chinese-instrument'`, `--flute-dir` → `--source-dir`
- Markdown cells: "flute" → "target instrument" contextually
- Output variable names: `flute_path` → `stem_path`, `flute_stems/` → `instrument_stems/`

**Step 2: Commit**

```bash
git add notebooks/
git commit -m "docs: update notebooks for instrument-agnostic naming"
```

---

## Phase 4: Final verification

### Task 11: Full test suite and cleanup

**Objective:** Verify no stale references remain and all 32 tests pass.

**Step 1: Search for remaining flute references**

```bash
grep -ri 'chinese.flute\|chinese-flute\|flute_dataset\|flute_dir\|flute_clip' \
    --include='*.py' --include='*.md' --include='*.yaml' --include='*.sh' \
    --include='*.ipynb' --include='*.toml' \
    --exclude-dir=vendor --exclude-dir=.venv --exclude-dir=.git .
```

Expected: zero matches outside `vendor/` and `.venv/`.

**Step 2: Run full test suite**

```bash
make test
```
Expected: 32 passed.

**Step 3: Run env-verify**

```bash
make env-verify
```
Expected: CUDA check + demucs import OK.

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: final cleanup — verify zero flute references remain"
```

---

## Files changed summary

| Category | Files |
|----------|-------|
| Rename | `chinese-flute-demucs/` → `chinese-instrument-demucs/` |
| Rename | `configs/dset/flute.yaml` → `instrument.yaml` |
| Rename | `configs/variant/flute_ft.yaml` → `instrument_ft.yaml` |
| Rename | `data_pipeline/validate_flute_free.py` → `validate_contamination.py` |
| Modify | `pyproject.toml` |
| Modify | `data_pipeline/build_dataset.py` |
| Modify | `inference/separate.py` |
| Modify | `eval/evaluate.py` |
| Modify | `training/patch_checkpoint.py` |
| Modify | `training/train.sh` |
| Modify | `tests/test_data_pipeline.py` |
| Modify | 10 docs files |
| Modify | 5 notebooks |
| Modify | `Makefile` (verify no conda/flute refs) |
| Modify | `README.md`, `mkdocs.yml`, `sample_data/README.md` |

## Risks & tradeoffs

- **Backward incompatibility:** Old flute-trained checkpoints have `'chinese-flute'` in the state dict. Users must retrain or manually rename the key. Document this in the migration guide (troubleshooting.md).
- **Config name change breaks Dora experiments:** `dset=flute` → `dset=instrument` means old experiment signatures won't resume. Acceptable — this is a new repo.
- **validate_contamination.py spectral heuristic:** Renamed from flute-band ratio but the frequency band (400-5000 Hz) is still tuned for midrange instruments. This works for most Chinese instruments (erhu, pipa, dizi, etc.) but document that users may need to adjust the band for very low (bass) or very high instruments.
