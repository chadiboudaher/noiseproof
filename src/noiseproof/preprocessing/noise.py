import torch

def calculate_power(waveform: torch.Tensor):
    signal_power = torch.mean(waveform ** 2)

    return signal_power


def match_noise_length(
        noise_waveform: torch.Tensor,
        target_length: int
):
    noise_sample = noise_waveform.shape[1]
    if noise_sample > target_length:
        noise_waveform = noise_waveform[:, :target_length]

    return noise_waveform