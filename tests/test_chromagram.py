import pytest
from pydantic import ValidationError

from chord_rec.chromagram import ChromagramConfig


class TestChromagramConfig:
    def test_defaults(self):
        config = ChromagramConfig()
        assert config.version == "STFT"
        assert config.sample_rate == 22050
        assert config.window_size == 4096
        assert config.hop_size == 2048
        assert config.gamma is None
        assert config.norm == "2"

    def test_valid_versions(self):
        for version in ("STFT", "CQT", "IIR"):
            config = ChromagramConfig(version=version)
            assert config.version == version

    def test_invalid_version_raises(self):
        with pytest.raises(ValidationError):
            ChromagramConfig(version="FFT")

    def test_gamma_must_be_positive(self):
        with pytest.raises(ValidationError):
            ChromagramConfig(gamma=0.0)
        with pytest.raises(ValidationError):
            ChromagramConfig(gamma=-1.0)

    def test_gamma_none_is_valid(self):
        config = ChromagramConfig(gamma=None)
        assert config.gamma is None

    def test_window_size_must_be_positive(self):
        with pytest.raises(ValidationError):
            ChromagramConfig(window_size=0)
        with pytest.raises(ValidationError):
            ChromagramConfig(window_size=-512)

    def test_hop_size_must_be_positive(self):
        with pytest.raises(ValidationError):
            ChromagramConfig(hop_size=0)

    def test_sample_rate_must_be_positive(self):
        with pytest.raises(ValidationError):
            ChromagramConfig(sample_rate=0)

    def test_valid_norm_values(self):
        for norm in ("1", "2", "max", None):
            config = ChromagramConfig(norm=norm)
            assert config.norm == norm

    def test_invalid_norm_raises(self):
        with pytest.raises(ValidationError):
            ChromagramConfig(norm="inf")
