"""Tests for the data pipeline: audio utilities, dataset builder, and verification."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from data_pipeline.audio_utils import (
    SAMPLE_RATE,
    ensure_length,
    ensure_stereo,
    loudness_normalize,
    mix_at_snr,
    pitch_shift,
    random_gain,
    rms,
    time_stretch,
)
from data_pipeline.build_dataset import build_dataset, build_track
from data_pipeline.validate_contamination import _instrument_band_ratio, validate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_file(path: Path, wav: torch.Tensor, sr: int = SAMPLE_RATE) -> Path:
    """Save a WAV and return its path."""
    import soundfile as sf
    # Convert (channels, samples) → (samples, channels) for soundfile
    sf.write(str(path), wav.numpy().T, sr, subtype="FLOAT")
    return path


def _white_noise(seconds: float = 1.0, stereo: bool = True) -> torch.Tensor:
    n = int(seconds * SAMPLE_RATE)
    ch = 2 if stereo else 1
    return torch.randn(ch, n) * 0.01


def _sine_tone(freq: float, seconds: float = 1.0, sr: int = SAMPLE_RATE) -> torch.Tensor:
    t = torch.arange(int(seconds * sr), dtype=torch.float32) / sr
    return (torch.sin(2 * np.pi * freq * t)).unsqueeze(0) * 0.1  # (1, n)


# ---------------------------------------------------------------------------
# audio_utils tests
# ---------------------------------------------------------------------------

class TestEnsureStereo:
    def test_mono_to_stereo(self):
        mono = torch.randn(1, 1000)
        out = ensure_stereo(mono)
        assert out.shape[0] == 2
        assert torch.equal(out[0], out[1])

    def test_stereo_passthrough(self):
        stereo = torch.randn(2, 1000)
        out = ensure_stereo(stereo)
        assert out.shape == stereo.shape

    def test_multichannel_truncate(self):
        multi = torch.randn(6, 1000)
        out = ensure_stereo(multi)
        assert out.shape[0] == 2


class TestEnsureLength:
    def test_crop(self):
        wav = torch.randn(2, 2000)
        out = ensure_length(wav, 1000)
        assert out.size(-1) == 1000

    def test_pad(self):
        wav = torch.randn(2, 500)
        out = ensure_length(wav, 1000)
        assert out.size(-1) == 1000


class TestRMS:
    def test_constant(self):
        wav = torch.ones(2, 1000) * 0.1
        assert abs(rms(wav).item() - 0.1) < 1e-6


class TestLoudnessNormalize:
    def test_target_db(self):
        wav = torch.ones(2, 1000) * 0.01
        out = loudness_normalize(wav, target_db=-20.0)
        # RMS should now be ~10^(-20/20) = 0.1
        assert abs(rms(out).item() - 0.1) < 0.01

    def test_silence_unchanged(self):
        wav = torch.zeros(2, 1000)
        out = loudness_normalize(wav, target_db=-20.0)
        assert torch.equal(out, wav)


class TestMixAtSNR:
    def test_positive_snr_signal_louder(self):
        # Use uncorrelated signals so powers add, not amplitudes
        torch.manual_seed(42)
        signal = torch.randn(2, 1000)
        other = torch.randn(2, 1000)
        mixture = mix_at_snr(signal, other, snr_db=10.0)
        # At SNR=10dB: P_other_scaled = P_signal / 10
        p_signal = signal.square().mean()
        p_other_scaled = (mixture - signal).square().mean()
        ratio = p_signal / p_other_scaled
        # Should be ~10 (10 dB)
        assert 8.0 < ratio.item() < 12.0

    def test_negative_snr_signal_quieter(self):
        torch.manual_seed(42)
        signal = torch.randn(2, 1000)
        other = torch.randn(2, 1000)
        mixture = mix_at_snr(signal, other, snr_db=-10.0)
        # At SNR=-10dB: P_other_scaled = P_signal * 10
        p_signal = signal.square().mean()
        p_other_scaled = (mixture - signal).square().mean()
        ratio = p_signal / p_other_scaled
        # Should be ~0.1 (-10 dB)
        assert 0.05 < ratio.item() < 0.2


class TestPitchShift:
    def test_no_shift_identity(self):
        wav = torch.randn(2, 1000)
        out = pitch_shift(wav, n_steps=0)
        assert torch.allclose(out, wav)

    def test_shift_preserves_length(self):
        wav = torch.randn(2, SAMPLE_RATE)
        out = pitch_shift(wav, n_steps=3)
        assert out.size(-1) == SAMPLE_RATE


class TestTimeStretch:
    def test_no_stretch_identity(self):
        wav = torch.randn(2, 1000)
        out = time_stretch(wav, rate=1.0)
        assert torch.allclose(out, wav)

    def test_stretch_changes_length(self):
        wav = torch.randn(2, SAMPLE_RATE)
        out = time_stretch(wav, rate=0.5)
        assert out.size(-1) == SAMPLE_RATE * 2


class TestRandomGain:
    def test_amplitude_change(self):
        wav = torch.ones(2, 1000)
        out = random_gain(wav, db_range=(0.0, 0.0))
        assert torch.allclose(out, wav)

    def test_range(self):
        wav = torch.ones(2, 1000)
        out = random_gain(wav, db_range=(-6.0, 6.0))
        assert not torch.allclose(out, wav) or abs(out.mean()) > 0


# ---------------------------------------------------------------------------
# build_track tests
# ---------------------------------------------------------------------------

class TestBuildTrack:
    def test_invariant_mixture_equals_sum(self, tmp_path: Path):
        """The plan's key invariant: mixture == source + other sample-exact."""
        # Create synthetic clips
        source_file = tmp_path / "source.wav"
        bg_file = tmp_path / "bg.wav"
        _make_wav_file(source_file, _sine_tone(440, seconds=3.0))
        _make_wav_file(bg_file, _white_noise(seconds=3.0))

        seg_samples = int(1.0 * SAMPLE_RATE)
        stems = build_track(
            source_file, bg_file,
            seg_samples=seg_samples, snr_db=5.0,
            augment_source=False, augment_bg=False,
        )

        mixture = stems["mixture"]
        default_source_name = "chinese-instrument"
        expected = stems[default_source_name] + stems["other"]

        max_diff = (mixture - expected).abs().max().item()
        assert max_diff < 1e-6, f"mixture ≠ source+other (max diff: {max_diff})"

    def test_output_is_stereo(self, tmp_path: Path):
        source_file = tmp_path / "source.wav"
        bg_file = tmp_path / "bg.wav"
        _make_wav_file(source_file, _sine_tone(440, seconds=3.0))
        _make_wav_file(bg_file, _white_noise(seconds=3.0))

        stems = build_track(
            source_file, bg_file,
            seg_samples=SAMPLE_RATE, snr_db=0.0,
            augment_source=False, augment_bg=False,
        )
        for name in ("mixture", "chinese-instrument", "other"):
            assert stems[name].size(0) == 2, f"{name} not stereo"
            assert stems[name].size(-1) == SAMPLE_RATE

    def test_output_is_non_silent(self, tmp_path: Path):
        source_file = tmp_path / "source.wav"
        bg_file = tmp_path / "bg.wav"
        _make_wav_file(source_file, _sine_tone(440, seconds=3.0))
        _make_wav_file(bg_file, _white_noise(seconds=3.0))

        stems = build_track(
            source_file, bg_file,
            seg_samples=SAMPLE_RATE, snr_db=0.0,
        )
        for name in ("mixture", "chinese-instrument", "other"):
            assert stems[name].abs().max() > 1e-6, f"{name} is silent"


# ---------------------------------------------------------------------------
# build_dataset integration tests
# ---------------------------------------------------------------------------

class TestBuildDataset:
    def test_creates_correct_structure(self, tmp_path: Path):
        source_dir = tmp_path / "source"
        bg_dir = tmp_path / "bg"
        out_dir = tmp_path / "dataset"
        for d in (source_dir, bg_dir):
            d.mkdir()

        # 5 source clips, 5 backgrounds — use short segments
        for i in range(5):
            _make_wav_file(source_dir / f"source_{i}.wav", _sine_tone(440 + i * 50, seconds=1.5))
            _make_wav_file(bg_dir / f"bg_{i}.wav", _white_noise(seconds=1.5))

        build_dataset(
            source_dir=source_dir,
            bg_dir=bg_dir,
            out_dir=out_dir,
            num_train=10,
            num_valid=3,
            seg_len=1.0,
            snr_min=0.0,
            snr_max=5.0,
            seed=99,
        )

        # Check structure
        source_name = "chinese-instrument"
        for split in ("train", "valid"):
            split_dir = out_dir / split
            assert split_dir.is_dir()

            tracks = sorted(split_dir.iterdir())
            for track_dir in tracks:
                assert (track_dir / "mixture.wav").is_file()
                assert (track_dir / f"{source_name}.wav").is_file()
                assert (track_dir / "other.wav").is_file()

                # Verify invariant on disk
                import soundfile as sf
                mix_data, sr = sf.read(str(track_dir / "mixture.wav"))
                source_data, _ = sf.read(str(track_dir / f"{source_name}.wav"))
                other_data, _ = sf.read(str(track_dir / "other.wav"))
                mix = torch.from_numpy(mix_data.T)
                source = torch.from_numpy(source_data.T)
                other = torch.from_numpy(other_data.T)
                assert sr == SAMPLE_RATE
                assert mix.size(0) == 2
                max_diff = (mix - (source + other)).abs().max().item()
                assert max_diff < 1e-6, f"Invariant violated in {track_dir}"

            assert len(tracks) == (10 if split == "train" else 3)

    def test_train_valid_no_shared_source_clips(self, tmp_path: Path):
        """Train/valid must share NO source clip identities."""
        source_dir = tmp_path / "source"
        bg_dir = tmp_path / "bg"
        out_dir = tmp_path / "dataset"
        source_dir.mkdir()
        bg_dir.mkdir()

        for i in range(3):
            _make_wav_file(source_dir / f"source_{i}.wav", _sine_tone(440 + i * 50, seconds=0.5))
        for i in range(3):
            _make_wav_file(bg_dir / f"bg_{i}.wav", _white_noise(seconds=0.5))

        build_dataset(
            source_dir=source_dir, bg_dir=bg_dir, out_dir=out_dir,
            num_train=2, num_valid=1, seg_len=0.2,
        )

        # Verify no identical source files appear in both splits
        source_name = "chinese-instrument"
        train_sources = []
        valid_sources = []
        import soundfile as sf
        for split, store in [("train", train_sources), ("valid", valid_sources)]:
            for track_dir in (out_dir / split).iterdir():
                data, _ = sf.read(str(track_dir / f"{source_name}.wav"))
                wav = torch.from_numpy(data.T)
                store.append(wav)

        # Every valid source should differ from every train source
        for vs in valid_sources:
            for ts in train_sources:
                assert not torch.allclose(vs, ts), "Train/valid share a source clip!"


# ---------------------------------------------------------------------------
# validate_contamination tests
# ---------------------------------------------------------------------------

class TestInstrumentBandRatio:
    def test_sine_vs_noise(self):
        """A sine in the instrument band should have a higher ratio than white noise."""
        sine = _sine_tone(1000, seconds=3.0)  # middle of instrument band
        noise = _white_noise(seconds=3.0, stereo=False)

        ratio_sine = _instrument_band_ratio(sine)
        ratio_noise = _instrument_band_ratio(noise)
        assert ratio_sine > ratio_noise, (
            f"Sine {ratio_sine:.3f} should have higher instrument-band ratio than noise {ratio_noise:.3f}"
        )

    def test_low_tone_low_ratio(self):
        """A 100 Hz tone (below instrument band) should have low ratio."""
        low = _sine_tone(100, seconds=3.0)
        ratio = _instrument_band_ratio(low)
        assert ratio < 0.3, f"Low tone ratio {ratio:.3f} should be low"


class TestValidate:
    def test_clean_backgrounds_no_flags(self, tmp_path: Path):
        bg_dir = tmp_path / "clean"
        bg_dir.mkdir()
        for i in range(5):
            _make_wav_file(bg_dir / f"bg_{i}.wav", _white_noise(seconds=2))
        flagged = validate(bg_dir, threshold=2.0)
        assert len(flagged) == 0

    def test_flags_instrument_like_files(self, tmp_path: Path):
        bg_dir = tmp_path / "mixed"
        bg_dir.mkdir()
        # Mostly clean backgrounds
        for i in range(4):
            _make_wav_file(bg_dir / f"bg_{i}.wav", _white_noise(seconds=2))
        # One instrument-like file
        _make_wav_file(bg_dir / f"suspicious.wav", _sine_tone(880, seconds=2))
        flagged = validate(bg_dir, abs_thresh=0.5)
        # The instrument-like one should be an outlier
        assert len(flagged) >= 1, "Should flag the instrument-like background"

    def test_output_json(self, tmp_path: Path):
        bg_dir = tmp_path / "clean"
        bg_dir.mkdir()
        _make_wav_file(bg_dir / "bg.wav", _white_noise(seconds=2))
        out_json = tmp_path / "results.json"
        validate(bg_dir, threshold=2.0, out_json=out_json)
        assert out_json.is_file()
        import json
        data = json.loads(out_json.read_text())
        assert isinstance(data, list)
