from pathlib import Path

import torch
import torchaudio

from noiseproof.preprocessing.audio import (
    AudioException,
    load_audio,
)


def create_fake_audio(
    tmp_path: Path,
    sample_rate: int = 8_000,
    channels: int = 1,
    duration_seconds: float = 1.0,
) -> Path:
    num_samples = int(sample_rate * duration_seconds)

    waveform = torch.randn(
        channels,
        num_samples,
    ) * 0.1

    audio_path = tmp_path / "test_audio.wav"

    torchaudio.save(
        audio_path,
        waveform,
        sample_rate,
    )

    return audio_path


def test_load_audio_resamples_to_16khz(tmp_path):
    audio_path = create_fake_audio(
        tmp_path=tmp_path,
        sample_rate=8_000,
        channels=1,
    )

    waveform, sample_rate = load_audio(audio_path)

    assert sample_rate == 16_000
    assert waveform.shape[0] == 1

    # 1 second at 16 kHz should contain approximately 16,000 samples.
    assert waveform.shape[1] == 16_000


def test_load_audio_keeps_16khz_audio(tmp_path):
    audio_path = create_fake_audio(
        tmp_path=tmp_path,
        sample_rate=16_000,
        channels=1,
    )

    waveform, sample_rate = load_audio(audio_path)

    assert sample_rate == 16_000
    assert waveform.shape == (1, 16_000)


def test_load_audio_converts_stereo_to_mono(tmp_path):
    audio_path = create_fake_audio(
        tmp_path=tmp_path,
        sample_rate=16_000,
        channels=2,
    )

    waveform, sample_rate = load_audio(audio_path)

    assert sample_rate == 16_000
    assert waveform.shape[0] == 1
    assert waveform.shape[1] == 16_000


def test_load_audio_raises_for_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.wav"

    try:
        load_audio(missing_path)

        assert False, "Expected AudioException"

    except AudioException:
        pass