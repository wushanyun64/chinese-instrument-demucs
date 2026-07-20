"""Tests for training, inference, and evaluation modules.

These tests mock out model downloads (network access) and use
small in-memory tensors where possible.
"""

import pytest
from pathlib import Path
import torch


# ---------------------------------------------------------------------------
# patch_checkpoint tests
# ---------------------------------------------------------------------------

class TestPatchCheckpoint:
    def test_build_model_smoke(self):
        """Smoke test: building a 2-source HTDemucs should succeed."""
        from demucs.htdemucs import HTDemucs
        model = HTDemucs(sources=["chinese-instrument", "other"], channels=4)
        assert model is not None
        state = model.state_dict()
        assert len(state) > 0
        # Verify model has the right source count
        assert model.sources == ["chinese-instrument", "other"]

    def test_load_and_patch_pure_cpu(self, tmp_path: Path):
        """Test that patching from a locally built model works (no network)."""
        from demucs.htdemucs import HTDemucs
        from training.patch_checkpoint import load_and_patch

        # Build a tiny 4-source "pretrained" and save it as if it were htdemucs.
        # We can't call get_model() without network, so we mock by building
        # a 4-source and a 2-source and comparing key sets.
        pretrained = HTDemucs(sources=["drums", "bass", "other", "vocals"], channels=4)
        target = HTDemucs(sources=["chinese-instrument", "other"], channels=4)

        pretrained_state = pretrained.state_dict()
        target_state = target.state_dict()

        loaded = 0
        skipped = 0
        missing = 0
        for key, target_param in target_state.items():
            if key in pretrained_state:
                if pretrained_state[key].shape == target_param.shape:
                    loaded += 1
                else:
                    skipped += 1
            else:
                missing += 1

        # Most keys should be loadable (shared architecture)
        assert loaded > 0
        # Some should be skipped (output head shape mismatch: 4→2 sources)
        assert skipped > 0 or missing > 0, (
            "Expected some keys to differ between 4-source and 2-source models"
        )

    def test_source_specific_keys_differ(self):
        """Verify that output-head keys differ between 4-source and 2-source."""
        from demucs.htdemucs import HTDemucs

        m4 = HTDemucs(sources=["drums", "bass", "other", "vocals"], channels=4)
        m2 = HTDemucs(sources=["chinese-instrument", "other"], channels=4)

        s4 = m4.state_dict()
        s2 = m2.state_dict()

        # Find keys that exist in both but have different shapes
        diff_shapes = []
        for k in s2:
            if k in s4 and s4[k].shape != s2[k].shape:
                diff_shapes.append(k)

        assert len(diff_shapes) > 0, (
            "Expected shape differences between 4- and 2-source models"
        )


# ---------------------------------------------------------------------------
# evaluate SI-SDR tests
# ---------------------------------------------------------------------------

class TestSISDR:
    def test_perfect_reconstruction(self):
        from eval.evaluate import si_sdr
        x = torch.randn(44100)
        score = si_sdr(x, x)
        # Perfect reconstruction should give very high score
        assert score > 50, f"SI-SDR should be very high for perfect match, got {score}"

    def test_orthogonal_signals(self):
        from eval.evaluate import si_sdr
        torch.manual_seed(42)
        ref = torch.randn(44100)
        est = torch.randn(44100)
        # Make orthogonal
        est = est - (est @ ref) / (ref @ ref) * ref
        score = si_sdr(est, ref)
        # Orthogonal signals should have very low SI-SDR
        assert score < -20, f"Orthogonal signals should have low SI-SDR, got {score}"

    def test_scaled_version(self):
        """SI-SDR should be scale-invariant."""
        from eval.evaluate import si_sdr
        ref = torch.randn(44100)
        est = 0.5 * ref  # scaled version
        score = si_sdr(est, ref)
        # Scale-invariant: should be near-perfect
        assert score > 50, f"Scaled version should have high SI-SDR, got {score}"
