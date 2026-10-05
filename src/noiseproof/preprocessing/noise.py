import math
import torch

class WaveformLengthMismatchError(Exception):
    pass

class SilentNoiseError(Exception):
    pass

class NoiseTooShortError(Exception):
    pass

def calculate_power(waveform: torch.Tensor):
    signal_power = torch.mean(waveform ** 2)

    return signal_power.item()


def match_noise_length(
        noise_waveform: torch.Tensor,
        target_length: int
):
    noise_length = noise_waveform.shape[1]
    if noise_length > target_length:
        noise_waveform = noise_waveform[:, :target_length]
    elif noise_length < target_length:
        raise NoiseTooShortError(
            f"noise sample length is smaller."
        )

    return noise_waveform


def calculate_noise_scale(
        speech_power,
        noise_power,
        target_snr_db
):
    return math.sqrt(
        speech_power / (noise_power * 10 ** (target_snr_db / 10))
    )

def calculate_snr(
        speech_waveform: torch.Tensor,
        noise_waveform: torch.Tensor
):
    if (
        speech_waveform.shape[1] != noise_waveform.shape[1]
    ):
        raise WaveformLengthMismatchError(
            f"Speech and noise waveforms must have the same length."
        )

    speech_power = calculate_power(speech_waveform)
    noise_power = calculate_power(noise_waveform)

    if noise_power == 0:
        raise SilentNoiseError(
            f"can't divide by zero."
        )

    snr_db = 10 * math.log10(speech_power / noise_power)

    return snr_db
