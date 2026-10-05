import pytest
import torch

from noiseproof.preprocessing.noise import (
    SilentNoiseError,
    NoiseTooShortError,
    WaveformLengthMismatchError,
    calculate_power,
    calculate_noise_scale,
    calculate_snr,
    match_noise_length,
)


def test_calculate_power():
    waveform = torch.tensor([[1.0, 2.0]])

    power = calculate_power(waveform)

    # (1^2 + 2^2) / 2 = 2.5
    assert power == pytest.approx(2.5)


def test_match_noise_length():
    noise = torch.randn(1, 1000)

    matched = match_noise_length(
        noise_waveform=noise,
        target_length=400,
    )

    assert matched.shape == (1, 400)


def test_match_noise_length_raises_if_noise_too_short():
    noise = torch.randn(1, 100)

    with pytest.raises(NoiseTooShortError):
        match_noise_length(
            noise_waveform=noise,
            target_length=200,
        )


def test_noise_scale_decreases_as_snr_increases():
    speech_power = 0.0025
    noise_power = 0.00000228

    scale_minus_5 = calculate_noise_scale(
        speech_power,
        noise_power,
        -5,
    )

    scale_0 = calculate_noise_scale(
        speech_power,
        noise_power,
        0,
    )

    scale_5 = calculate_noise_scale(
        speech_power,
        noise_power,
        5,
    )

    scale_10 = calculate_noise_scale(
        speech_power,
        noise_power,
        10,
    )

    assert scale_minus_5 > scale_0 > scale_5 > scale_10


@pytest.mark.parametrize(
    "target_snr_db",
    [-5, 0, 5, 10],
)
def test_scaled_noise_reaches_target_snr(target_snr_db):
    speech = torch.randn(1, 16000)
    noise = torch.randn(1, 16000)

    speech_power = calculate_power(speech)
    noise_power = calculate_power(noise)

    alpha = calculate_noise_scale(
        speech_power=speech_power,
        noise_power=noise_power,
        target_snr_db=target_snr_db,
    )

    scaled_noise = noise * alpha

    achieved_snr = calculate_snr(
        speech_waveform=speech,
        noise_waveform=scaled_noise,
    )

    assert achieved_snr == pytest.approx(
        target_snr_db,
        abs=1e-4,
    )


def test_calculate_snr_raises_for_length_mismatch():
    speech = torch.randn(1, 1000)
    noise = torch.randn(1, 500)

    with pytest.raises(WaveformLengthMismatchError):
        calculate_snr(
            speech_waveform=speech,
            noise_waveform=noise,
        )


def test_calculate_snr_raises_for_silent_noise():
    speech = torch.randn(1, 1000)
    noise = torch.zeros(1, 1000)

    with pytest.raises(SilentNoiseError):
        calculate_snr(
            speech_waveform=speech,
            noise_waveform=noise,
        )