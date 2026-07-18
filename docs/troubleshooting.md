# Troubleshooting

## Strict checkpoint load crash

**Symptom:** `RuntimeError: size mismatch` when loading pretrained htdemucs.

**Cause:** 4→2 source shape mismatch in the output head.

**Fix:** Use `training/patch_checkpoint.py` — loads with `strict=False`, skips
shape-mismatched keys, logs which ones were skipped.

## Stems not summing to mixture

**Symptom:** Training loss oscillates or doesn't converge.

**Cause:** `other.wav` was recomputed lossily after writing `chinese-instrument.wav`.

**Fix:** Write `other.wav` as the literal background, then write `mixture.wav` as
`chinese-instrument.wav + other.wav` sample-exact. Do not recompute.

## Contaminated backgrounds

**Symptom:** Model leaves target instrument in `other` output.

**Cause:** Background audio contains target instrument, teaching the model target instrument belongs in `other`.

**Fix:** Run `data_pipeline/validate_target instrument_free.py` on your backgrounds and remove flagged files.

## Overfitting to narrow target instrument data

**Symptom:** Great on synthetic validation, fails on real music.

**Cause:** All target instrument clips share the same instrument/player/recording condition.

**Fix:** Diversify target instrument clips — multiple instruments (dizi/xiao/bawu), players, keys,
articulations, and recording conditions.

## OOM (Out of Memory)

**Symptom:** `CUDA out of memory` during training.

**Fix:** Reduce `dset.segment` and/or `dset.batch_size` in `configs/variant/target instrument_ft.yaml`.

## Silent / NaN audio

**Symptom:** Output wav is silent or contains NaN samples.

**Cause:** Gain/normalization producing extreme values, or corrupt input audio.

**Fix:** Check input clips for silent/corrupt sections. Verify loudness normalization parameters.

## Sample rate mismatches

**Symptom:** Audio sounds sped up/slowed down.

**Cause:** Input not resampled to 44.1 kHz.

**Fix:** `audio_utils.py` resamples all inputs to 44.1 kHz before mixing.
